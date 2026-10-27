"""Real MyTorch ResNet18 worker for the gRPC parameter server.

Unlike ``grpc_worker.py`` (a transport smoke test), this worker performs the
actual MyTorch forward/backward path and sends flattened ``Tensor.grad`` data
to the server after every batch.
"""
from __future__ import annotations

import argparse
import time

import grpc
import numpy as np

from model.resnet import ResNet18Original
from mytorch.dataset import AutoDriveDataset
from mytorch.dataloader import Dataloader
from mytorch.loss import MSELoss

from . import training_pb2, training_pb2_grpc
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


def run(args) -> None:
    model = ResNet18Original(num_classes=1).train()
    if args.device == "cuda":
        model.cuda()
    parameter_count = int(flatten_parameters(model).size)
    if args.print_parameter_count:
        print(parameter_count)
        return

    channel = grpc.insecure_channel(
        args.target,
        options=[("grpc.max_send_message_length", 128 * 1024 * 1024), ("grpc.max_receive_message_length", 128 * 1024 * 1024)],
    )
    stub = training_pb2_grpc.ParameterServerStub(channel)
    base = AutoDriveDataset("train", data_root=args.data_root, list_path=args.train_list)
    indices = np.arange(len(base))[args.worker_id::args.workers]
    loader = Dataloader(IndexedDataset(base, indices), batch_size=args.batch_size, shuffle=True)
    criterion = MSELoss(fused=False)
    rows = []

    for epoch in range(args.epochs):
        total_epoch = Stopwatch()
        compute_ms = 0.0
        communication_ms = 0.0
        message_bytes = 0
        seen = 0
        accuracy_sum = 0.0
        accuracy_count = 0
        accumulated_gradient = None
        accumulated_samples = 0
        accumulation_count = 0
        pulled_parameters = False
        for batch_index, (x, y) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            if not pulled_parameters:
                pull_clock = Stopwatch()
                pulled = stub.Pull(training_pb2.PullRequest(worker_id=f"worker-{args.worker_id}", known_version=0))
                communication_ms += pull_clock.elapsed_ms()
                load_parameters(model, np.frombuffer(pulled.parameters, dtype=np.float32).copy())
                pulled_parameters = True

            compute = Stopwatch()
            if args.device == "cuda":
                x = x.cuda()
                y = y.cuda()
            model.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            gradient = flatten_gradients(model)
            if accumulated_gradient is None:
                accumulated_gradient = np.zeros_like(gradient, dtype=np.float32)
            accumulated_gradient += gradient
            accumulated_samples += int(x.shape()[0])
            accumulation_count += 1
            compute_ms += compute.elapsed_ms()
            prediction_cpu = prediction.data.get() if hasattr(prediction.data, "get") else np.asarray(prediction.data)
            target_cpu = y.data.get() if hasattr(y.data, "get") else np.asarray(y.data)
            batch_accuracy = float(np.mean(np.abs(prediction_cpu - target_cpu) < args.accuracy_threshold))

            is_last = batch_index + 1 >= len(loader)
            if accumulation_count >= args.accumulation_steps or is_last:
                average_gradient = accumulated_gradient / float(accumulation_count)
                send_gradient = average_gradient.astype(np.float16 if args.fp16_gradients else np.float32)
                push_clock = Stopwatch()
                pushed = stub.Push(training_pb2.PushRequest(worker_id=f"worker-{args.worker_id}", base_version=pulled.version if pulled_parameters else 0, gradient=send_gradient.tobytes(), gradient_dtype="float16" if args.fp16_gradients else "float32", sample_count=accumulated_samples, compute_ms=compute_ms, accuracy=batch_accuracy))
                communication_ms += push_clock.elapsed_ms()
                message_bytes += int(pushed.message_bytes)
                load_parameters(model, np.frombuffer(pushed.parameters, dtype=np.float32).copy())
                accumulated_gradient = None
                accumulated_samples = 0
                accumulation_count = 0
                pulled_parameters = False
            seen += int(x.shape()[0])
            accuracy_sum += batch_accuracy * int(x.shape()[0])
            accuracy_count += int(x.shape()[0])

        rows.append(EpochMetrics("grpc-protobuf-mytorch-resnet18", args.workers, epoch, total_epoch.elapsed_ms(), compute_ms, communication_ms, message_bytes, accuracy_sum / accuracy_count if accuracy_count else 0.0))

    stub.Finish(training_pb2.FinishRequest(worker_id=f"worker-{args.worker_id}"))
    write_metrics(rows, args.output)
    print(f"worker={args.worker_id} parameters={parameter_count} samples={seen} output={args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--accuracy-threshold", type=float, default=0.05)
    parser.add_argument("--output", default="results/grpc_mytorch_worker.csv")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--fp16-gradients", action="store_true")
    parser.add_argument("--print-parameter-count", action="store_true")
    run(parser.parse_args())
