"""Minimal parameter-server implementation using gRPC + protobuf.

The payload is deliberately model-agnostic: MyTorch workers can serialize their
parameter/gradient arrays with ``numpy.tobytes()`` and reuse this service.
"""
from __future__ import annotations

import argparse
import threading
from concurrent import futures

import numpy as np
import grpc

from . import training_pb2, training_pb2_grpc


class ParameterServerServicer(training_pb2_grpc.ParameterServerServicer):
    def __init__(self, parameter_count: int, learning_rate: float, initial_parameters: str | None = None) -> None:
        self.parameters = np.zeros(parameter_count, dtype=np.float32)
        if initial_parameters:
            self.parameters = np.asarray(np.load(initial_parameters), dtype=np.float32).reshape(-1)
            if self.parameters.size != parameter_count:
                raise ValueError("initial parameter count does not match server configuration")
        self.learning_rate = learning_rate
        self.version = 0
        self.lock = threading.Lock()
        self.finished: set[str] = set()

    def Pull(self, request, context):  # noqa: N802 - generated gRPC API
        with self.lock:
            return training_pb2.PullReply(parameters=self.parameters.tobytes(), version=self.version)

    def Push(self, request, context):  # noqa: N802 - generated gRPC API
        if request.gradient_dtype == "float16":
            gradient = np.frombuffer(request.gradient, dtype=np.float16).astype(np.float32)
        else:
            gradient = np.frombuffer(request.gradient, dtype=np.float32)
        with self.lock:
            if gradient.size != self.parameters.size:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "gradient size does not match server parameters")
            self.parameters -= self.learning_rate * gradient
            self.version += 1
            payload = self.parameters.tobytes()
            return training_pb2.PushReply(parameters=payload, version=self.version, message_bytes=len(request.gradient) + len(payload))

    def Finish(self, request, context):  # noqa: N802 - generated gRPC API
        with self.lock:
            self.finished.add(request.worker_id)
        return training_pb2.FinishReply(ok=True)


def serve(host: str, port: int, parameter_count: int, learning_rate: float, workers: int, initial_parameters: str | None = None) -> None:
    options = [("grpc.max_send_message_length", 128 * 1024 * 1024), ("grpc.max_receive_message_length", 128 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max(4, workers * 2)), options=options)
    service = ParameterServerServicer(parameter_count, learning_rate, initial_parameters)
    training_pb2_grpc.add_ParameterServerServicer_to_server(service, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"parameter-server listening on {host}:{port}; parameters={parameter_count}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--parameter-count", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--initial-parameters")
    args = parser.parse_args()
    serve(args.host, args.port, args.parameter_count, args.learning_rate, args.workers, args.initial_parameters)
