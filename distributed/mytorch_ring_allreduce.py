"""Complete MyTorch DonkeyCar CNN training with TCP Ring AllReduce.

Each rank owns a data shard and a full local model.  Gradients are averaged
with the educational Reduce-Scatter + All-Gather ring and parameters are
updated locally, so there is no central parameter server.
"""
from __future__ import annotations

import argparse
import base64
from concurrent import futures
import json
import multiprocessing as mp
import queue
import socket
import struct
import time
from pathlib import Path

import numpy as np

from model.resnet import ResNet18Original
from mytorch.dataset import AutoDriveDataset
from mytorch.dataloader import Dataloader
from mytorch.loss import MSELoss

from .metrics import EpochMetrics, Stopwatch, write_metrics
from .mytorch_adapter import flatten_gradients, flatten_parameters, load_parameters


class IndexedDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("ring peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_array(sock, array: np.ndarray, transport: str = "binary") -> int:
    if transport == "grpc_protobuf":
        return sock.send_array(array)
    payload = np.ascontiguousarray(array, dtype=np.float16).tobytes()
    if transport == "socket_json":
        payload = json.dumps({"dtype": "float16", "data": base64.b64encode(payload).decode("ascii")}, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!Q", len(payload)) + payload)
    return 8 + len(payload)


def _recv_array(sock, transport: str = "binary") -> tuple[np.ndarray, int]:
    if transport == "grpc_protobuf":
        return sock.recv_array()
    size = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    payload = _recv_exact(sock, size)
    if transport == "socket_json":
        message = json.loads(payload)
        if message.get("dtype") != "float16":
            raise ValueError("unexpected ring JSON dtype")
        payload = base64.b64decode(message["data"], validate=True)
    return np.frombuffer(payload, dtype=np.float16).astype(np.float32), 8 + size


class _GrpcSendLink:
    def __init__(self, channel):
        from google.protobuf.wrappers_pb2 import BytesValue

        self.channel = channel
        self.call = channel.unary_unary(
            "/ring.Gradient/Send",
            request_serializer=BytesValue.SerializeToString,
            response_deserializer=BytesValue.FromString,
        )

    def send_array(self, array: np.ndarray) -> int:
        from google.protobuf.wrappers_pb2 import BytesValue

        request = BytesValue(value=np.ascontiguousarray(array, dtype=np.float16).tobytes())
        self.call(request, timeout=120)
        return request.ByteSize()

    def close(self) -> None:
        self.channel.close()


class _GrpcRecvLink:
    def __init__(self, received: queue.Queue, server, timeout: float):
        self.received = received
        self.server = server
        self.timeout = timeout

    def recv_array(self) -> tuple[np.ndarray, int]:
        payload = self.received.get(timeout=self.timeout)
        from google.protobuf.wrappers_pb2 import BytesValue

        return np.frombuffer(payload, dtype=np.float16).astype(np.float32), BytesValue(value=payload).ByteSize()

    def close(self) -> None:
        self.server.stop(grace=1).wait(timeout=5)


def _ring_grpc_links(rank: int, workers: int, host: str, base_port: int, timeout: float):
    import grpc
    from google.protobuf.wrappers_pb2 import BytesValue

    received: queue.Queue[bytes] = queue.Queue()

    def accept(request: BytesValue, context) -> BytesValue:
        received.put(request.value)
        return BytesValue()

    handler = grpc.method_handlers_generic_handler(
        "ring.Gradient",
        {"Send": grpc.unary_unary_rpc_method_handler(
            accept,
            request_deserializer=BytesValue.FromString,
            response_serializer=BytesValue.SerializeToString,
        )},
    )
    options = [("grpc.max_send_message_length", 256 * 1024 * 1024),
               ("grpc.max_receive_message_length", 256 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2), options=options)
    server.add_generic_rpc_handlers((handler,))
    address = f"{host}:{base_port + rank}"
    if server.add_insecure_port(address) == 0:
        raise OSError(f"could not bind gRPC ring listener {address}")
    server.start()
    channel = grpc.insecure_channel(f"{host}:{base_port + ((rank + 1) % workers)}", options=options)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except Exception:
        channel.close()
        server.stop(grace=0)
        raise
    return _GrpcSendLink(channel), _GrpcRecvLink(received, server, timeout)


def _connect(host: str, port: int, timeout: float) -> socket.socket:
    deadline = time.time() + timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.settimeout(timeout)
            return sock
        except OSError:
            if time.time() >= deadline:
                raise
            time.sleep(0.2)


def _ring_sockets(rank: int, workers: int, host: str, base_port: int, timeout: float):
    # Receive from previous rank and send to next rank over separate TCP links.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, base_port + rank))
    listener.listen(1)
    send_sock = _connect(host, base_port + ((rank + 1) % workers), timeout)
    recv_sock, _ = listener.accept()
    recv_sock.settimeout(timeout)
    listener.close()
    return send_sock, recv_sock


def ring_allreduce(gradient: np.ndarray, rank: int, workers: int, send_sock, recv_sock, transport: str = "binary"):
    chunks = np.array_split(np.asarray(gradient, dtype=np.float32), workers)
    sent_bytes = 0
    received_bytes = 0

    # Reduce-Scatter: after this phase each rank owns one fully reduced chunk.
    for step in range(workers - 1):
        send_idx = (rank - step) % workers
        recv_idx = (rank - step - 1) % workers
        # Alternate order prevents both ends from filling TCP send buffers
        # with a large tensor before either side receives.
        if rank % 2 == 0:
            sent_bytes += _send_array(send_sock, chunks[send_idx], transport)
            incoming, nbytes = _recv_array(recv_sock, transport)
        else:
            incoming, nbytes = _recv_array(recv_sock, transport)
            sent_bytes += _send_array(send_sock, chunks[send_idx], transport)
        chunks[recv_idx] += incoming
        received_bytes += nbytes

    # All-Gather: circulate the reduced chunks until every rank has all chunks.
    owned = (rank - (workers - 1)) % workers
    for step in range(workers - 1):
        send_idx = (owned - step) % workers
        recv_idx = (owned - step - 1) % workers
        if rank % 2 == 0:
            sent_bytes += _send_array(send_sock, chunks[send_idx], transport)
            incoming, nbytes = _recv_array(recv_sock, transport)
        else:
            incoming, nbytes = _recv_array(recv_sock, transport)
            sent_bytes += _send_array(send_sock, chunks[send_idx], transport)
        chunks[recv_idx] = incoming
        received_bytes += nbytes

    result = np.concatenate(chunks) / float(workers)
    return result, sent_bytes + received_bytes


def _worker(rank: int, args) -> None:
    if args.device == "cuda":
        import cupy as cp

        gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank
        cp.cuda.Device(gpu_id).use()
        print(f"rank={rank} gpu_id={gpu_id}", flush=True)
    model = ResNet18Original(num_classes=1).train()
    initial = np.asarray(np.load(args.initial_parameters), dtype=np.float32).reshape(-1)
    load_parameters(model, initial)
    if args.device == "cuda":
        model.cuda()
    base = AutoDriveDataset("train", data_root=args.data_root, list_path=args.train_list)
    indices = np.arange(len(base))[rank :: args.workers]
    loader = Dataloader(IndexedDataset(base, indices), batch_size=args.batch_size, shuffle=True)
    criterion = MSELoss(fused=False)
    send_sock, recv_sock = (_ring_grpc_links(rank, args.workers, args.host, args.base_port, args.timeout)
                            if args.transport == "grpc_protobuf" else
                            _ring_sockets(rank, args.workers, args.host, args.base_port, args.timeout))
    rows = []

    for epoch in range(args.epochs):
        total_epoch = Stopwatch()
        compute_ms = 0.0
        communication_ms = 0.0
        message_bytes = 0
        seen = 0
        accuracy_sum = 0.0
        accuracy_count = 0
        accumulated = None
        accumulation_count = 0

        for batch_index, (x, y) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            compute = Stopwatch()
            if args.device == "cuda":
                x = x.cuda()
                y = y.cuda()
            model.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            gradient = flatten_gradients(model)
            if accumulated is None:
                accumulated = np.zeros_like(gradient, dtype=np.float32)
            accumulated += gradient
            accumulation_count += 1
            compute_ms += compute.elapsed_ms()
            prediction_cpu = prediction.data.get() if hasattr(prediction.data, "get") else np.asarray(prediction.data)
            target_cpu = y.data.get() if hasattr(y.data, "get") else np.asarray(y.data)
            batch_accuracy = float(np.mean(np.abs(prediction_cpu - target_cpu) < args.accuracy_threshold))
            seen += int(x.shape()[0])
            accuracy_sum += batch_accuracy * int(x.shape()[0])
            accuracy_count += int(x.shape()[0])

            is_last = batch_index + 1 >= len(loader)
            if accumulation_count >= args.accumulation_steps or is_last:
                average_local = accumulated / float(accumulation_count)
                clock = Stopwatch()
                average_global, wire_bytes = ring_allreduce(average_local, rank, args.workers, send_sock, recv_sock, args.transport)
                communication_ms += clock.elapsed_ms()
                message_bytes += wire_bytes
                current = flatten_parameters(model)
                load_parameters(model, current - args.learning_rate * average_global)
                accumulated = None
                accumulation_count = 0

        row = EpochMetrics("ring-allreduce-mytorch-resnet18", args.workers, epoch, total_epoch.elapsed_ms(), compute_ms, communication_ms, message_bytes, accuracy_sum / accuracy_count if accuracy_count else 0.0)
        rows.append(row)
        write_metrics(rows, str(Path(args.output_dir) / f"ring_worker{rank}.csv"))
        print(f"rank={rank} epoch={epoch + 1:02d} total_ms={row.total_ms:.1f} compute_ms={row.compute_ms:.1f} communication_ms={row.communication_ms:.1f} message_bytes={row.message_bytes} accuracy={row.accuracy:.4f}", flush=True)

    send_sock.close()
    recv_sock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--transport", choices=("binary", "socket_json", "grpc_protobuf"), default="binary")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--initial-parameters", required=True)
    parser.add_argument("--output-dir", default="results/ring_allreduce")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=50200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-ids", type=lambda value: [int(item) for item in value.split(",")], default=None,
                        help="Comma-separated physical GPU IDs, one for each rank (default: 0..workers-1)")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--accuracy-threshold", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.device == "cuda" and args.gpu_ids is not None and len(args.gpu_ids) != args.workers:
        parser.error("--gpu-ids must contain exactly one ID per worker")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    processes = [mp.Process(target=_worker, args=(rank, args), daemon=False) for rank in range(args.workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        raise SystemExit("one or more Ring AllReduce workers failed")


if __name__ == "__main__":
    main()
