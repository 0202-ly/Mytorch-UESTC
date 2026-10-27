"""Socket + JSON baseline matching the course comparison table."""
from __future__ import annotations

import argparse
import base64
import json
import socket
import socketserver
import threading
import time
from pathlib import Path

import numpy as np

from .metrics import EpochMetrics, Stopwatch, write_metrics


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        service: "JsonParameterServer" = self.server.service  # type: ignore[attr-defined]
        for line in self.rfile:
            request = json.loads(line)
            response = service.step(request)
            self.wfile.write((json.dumps(response) + "\n").encode())
            self.wfile.flush()


class JsonParameterServer:
    def __init__(self, parameter_count: int, learning_rate: float, initial_parameters: str | None = None) -> None:
        self.parameters = np.zeros(parameter_count, dtype=np.float32)
        if initial_parameters:
            self.parameters = np.asarray(np.load(initial_parameters), dtype=np.float32).reshape(-1)
            if self.parameters.size != parameter_count:
                raise ValueError("initial parameter count does not match server configuration")
        self.learning_rate = learning_rate
        self.lock = threading.Lock()

    def step(self, request: dict) -> dict:
        if request.get("op") == "pull":
            with self.lock:
                payload = self.parameters.tobytes()
            return {
                "op": "pull_reply",
                "parameters_b64": base64.b64encode(payload).decode("ascii"),
                "version": 0,
                "message_bytes": len(payload),
            }
        if request.get("op") == "push":
            gradient_bytes = base64.b64decode(request["gradient_b64"])
            gradient_dtype = request.get("gradient_dtype", "float32")
            if gradient_dtype == "float16":
                gradient = np.frombuffer(gradient_bytes, dtype=np.float16).astype(np.float32)
            elif gradient_dtype == "float32":
                gradient = np.frombuffer(gradient_bytes, dtype=np.float32)
            else:
                raise ValueError(f"unsupported gradient dtype: {gradient_dtype}")
            with self.lock:
                if gradient.size != self.parameters.size:
                    raise ValueError("gradient size does not match server parameters")
                self.parameters -= self.learning_rate * gradient
                payload = self.parameters.tobytes()
            return {
                "op": "push_reply",
                "parameters_b64": base64.b64encode(payload).decode("ascii"),
                "version": 0,
                "message_bytes": len(request["gradient_b64"]) + len(payload),
            }
        gradient = np.asarray(request["gradient"], dtype=np.float32)
        with self.lock:
            self.parameters -= self.learning_rate * gradient
            payload = self.parameters.tolist()
        return {"parameters": payload, "message_bytes": len(json.dumps(request).encode()) + len(json.dumps(payload).encode())}


class ThreadingJsonServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def serve(host: str, port: int, parameter_count: int, learning_rate: float, initial_parameters: str | None = None) -> None:
    service = JsonParameterServer(parameter_count, learning_rate, initial_parameters)
    with ThreadingJsonServer((host, port), _Handler) as server:
        server.service = service  # type: ignore[attr-defined]
        print(f"json parameter-server listening on {host}:{port}")
        server.serve_forever()


def run_worker(target: tuple[str, int], worker_id: str, epochs: int, parameter_count: int, output: str) -> None:
    rng = np.random.default_rng(abs(hash(worker_id)) % (2**32))
    rows: list[EpochMetrics] = []
    with socket.create_connection(target) as sock:
        reader = sock.makefile("rb")
        for epoch in range(epochs):
            total = Stopwatch()
            compute = Stopwatch()
            gradient = (rng.normal(size=parameter_count) * -1).astype(np.float32)
            compute_ms = compute.elapsed_ms()
            request = {"worker_id": worker_id, "gradient": gradient.tolist()}
            communication = Stopwatch()
            sock.sendall((json.dumps(request) + "\n").encode())
            response = json.loads(reader.readline())
            communication_ms = communication.elapsed_ms()
            rows.append(EpochMetrics("socket-json-ps", 1, epoch, total.elapsed_ms(), compute_ms, communication_ms, response["message_bytes"], float(np.mean(np.abs(gradient) < 1.0))))
    write_metrics(rows, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    server = sub.add_parser("server")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=50052)
    server.add_argument("--parameter-count", type=int, default=1024)
    server.add_argument("--learning-rate", type=float, default=0.01)
    server.add_argument("--initial-parameters")
    worker = sub.add_parser("worker")
    worker.add_argument("--host", default="127.0.0.1")
    worker.add_argument("--port", type=int, default=50052)
    worker.add_argument("--worker-id", default="worker-0")
    worker.add_argument("--epochs", type=int, default=5)
    worker.add_argument("--parameter-count", type=int, default=1024)
    worker.add_argument("--output", default="results/socket_worker.csv")
    args = parser.parse_args()
    if args.mode == "server":
        serve(args.host, args.port, args.parameter_count, args.learning_rate, args.initial_parameters)
    else:
        run_worker((args.host, args.port), args.worker_id, args.epochs, args.parameter_count, args.output)
