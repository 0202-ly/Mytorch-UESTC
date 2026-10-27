"""Repeat full DonkeyCar CNN training with one dedicated GPU per worker."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import numpy as np


def wait_for_server(host: str, port: int, process: subprocess.Popen, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"parameter server exited with {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"parameter server did not listen on {host}:{port}")


def gpu_monitor(path: Path, done: threading.Event) -> None:
    fields = "index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "gpu_index", "utilization_pct", "memory_used_mib",
                         "memory_total_mib", "temperature_c", "power_w"])
        while not done.is_set():
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            )
            stamp = time.time()
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    writer.writerow([stamp, *[part.strip() for part in line.split(",")]])
                handle.flush()
            done.wait(2)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_mode(args: argparse.Namespace, mode: str) -> dict:
    workers = int(mode[-1])
    if workers > args.available_gpus:
        raise RuntimeError(f"{mode} requires {workers} GPUs, only {args.available_gpus} available")
    folder = args.output_dir / mode
    folder.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    common = ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
              "--accumulation-steps", str(args.accumulation_steps),
              "--data-root", ".", "--train-list", "train.txt", "--device", "cuda"]
    if args.max_batches is not None:
        common += ["--max-batches", str(args.max_batches)]
    subprocesses: list[subprocess.Popen] = []
    open_logs = []
    done = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(folder / "gpu_monitor.csv", done), daemon=True)
    started = time.monotonic()
    try:
        monitor.start()
        if mode.startswith(("socket", "grpc")):
            parameter_count = int(np.load(args.initial_parameters, mmap_mode="r").size)
            port = args.base_port + workers + (20 if mode.startswith("grpc") else 0)
            server_log = (folder / "server.log").open("w", encoding="utf-8")
            open_logs.append(server_log)
            server_env = os.environ.copy()
            server_env["CUDA_VISIBLE_DEVICES"] = ""
            if mode.startswith("grpc"):
                server_command = [python, "-u", "-m", "distributed.parameter_server",
                                  "--host", "127.0.0.1", "--port", str(port),
                                  "--workers", str(workers)]
            else:
                server_command = [python, "-u", "-m", "distributed.socket_ps", "server",
                                  "--host", "127.0.0.1", "--port", str(port)]
            server_command += ["--parameter-count", str(parameter_count),
                               "--learning-rate", str(args.learning_rate),
                               "--initial-parameters", str(args.initial_parameters)]
            server = subprocess.Popen(
                server_command,
                stdout=server_log, stderr=subprocess.STDOUT, env=server_env,
            )
            subprocesses.append(server)
            wait_for_server("127.0.0.1", port, server)
            worker_processes = []
            for rank in range(workers):
                log = (folder / f"worker{rank}.log").open("w", encoding="utf-8")
                open_logs.append(log)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(rank)
                command = [python, "-u", "-m", ("distributed.mytorch_grpc_worker"
                           if mode.startswith("grpc") else "distributed.socket_mytorch_worker")]
                if mode.startswith("grpc"):
                    command += ["--target", f"127.0.0.1:{port}"]
                else:
                    command += ["--host", "127.0.0.1", "--port", str(port)]
                command += ["--workers", str(workers), "--worker-id", str(rank),
                            "--fp16-gradients", "--output", str(folder / f"worker{rank}.csv"),
                            *common]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                worker_processes.append(process)
                subprocesses.append(process)
            while any(process.poll() is None for process in worker_processes):
                if any(process.poll() not in (None, 0) for process in worker_processes):
                    raise RuntimeError("a Socket worker failed; check worker logs")
                if server.poll() is not None:
                    raise RuntimeError("parameter server stopped before the workers")
                time.sleep(2)
            if any(process.returncode != 0 for process in worker_processes):
                raise RuntimeError("a Socket worker failed; check worker logs")
        else:
            log = (folder / "ring.log").open("w", encoding="utf-8")
            open_logs.append(log)
            ring_transport = ("socket_json" if mode.startswith("ring_socket") else
                              "grpc_protobuf" if mode.startswith("ring_grpc") else "binary")
            command = [python, "-u", "-m", "distributed.mytorch_ring_allreduce",
                       "--workers", str(workers), "--learning-rate", str(args.learning_rate),
                       "--transport", ring_transport,
                       "--gpu-ids", ",".join(str(rank) for rank in range(workers)),
                       "--initial-parameters", str(args.initial_parameters),
                       "--output-dir", str(folder), "--base-port", str(args.base_port + 100 + workers * 10),
                       *common]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            subprocesses.append(process)
            if process.wait() != 0:
                raise RuntimeError("Ring training failed; check ring.log")
        result = {"mode": mode, "workers": workers, "gpu_ids": list(range(workers)),
                  "epochs": args.epochs, "batch_size": args.batch_size,
                  "accumulation_steps": args.accumulation_steps, "gradient_wire_dtype": "float16",
                  "max_batches": args.max_batches, "wall_seconds": time.monotonic() - started,
                  "status": "complete"}
        (folder / "run_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        for process in reversed(subprocesses):
            stop_process(process)
        done.set()
        if monitor.ident is not None:
            monitor.join(timeout=5)
        for handle in open_logs:
            handle.close()


def main() -> None:
    # CuPy needs the toolkit headers for its first just-in-time kernel build.
    if "CUDA_PATH" not in os.environ and Path("/usr/local/cuda-12.4").exists():
        os.environ["CUDA_PATH"] = "/usr/local/cuda-12.4"
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.4")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-parameters", type=Path,
                        default=Path("results/full_course/init_params.npy"))
    parser.add_argument("--modes", default="socket2,socket4,grpc2,grpc4,ring2,ring4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--base-port", type=int, default=51400)
    parser.add_argument("--available-gpus", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = args.modes.split(",")
    valid = {"socket2", "socket4", "grpc2", "grpc4", "ring2", "ring4",
             "ring_socket2", "ring_socket4", "ring_grpc2", "ring_grpc4"}
    if any(mode not in valid for mode in modes):
        parser.error("--modes contains an unsupported configuration")
    for mode in modes:
        print(f"START {mode} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        result = run_mode(args, mode)
        print(f"DONE {mode} {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
