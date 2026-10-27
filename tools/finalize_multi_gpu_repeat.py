"""Finish a multi-GPU repeat with a matched single-GPU run and figures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


MODES = ("socket2", "socket4", "grpc2", "grpc4", "ring2", "ring4")


def write_status(path: Path, phase: str, **details) -> None:
    path.write_text(json.dumps({"phase": phase, "timestamp": time.time(), **details}, indent=2), encoding="utf-8")


def launcher_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status = args.run_dir / "finalizer_status.json"
    write_status(status, "waiting_for_six_distributed_runs")
    try:
        while True:
            complete = all((args.run_dir / mode / "run_metadata.json").exists() for mode in MODES)
            alive = launcher_alive(args.launcher_pid)
            if complete and not alive:
                break
            if not alive:
                raise RuntimeError("experiment launcher exited before completing all six modes")
            time.sleep(30)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env.setdefault("CUDA_PATH", "/usr/local/cuda-12.4")
        env.setdefault("CUDA_HOME", "/usr/local/cuda-12.4")
        baseline = args.run_dir / "single_gpu_same_initial.csv"
        write_status(status, "single_gpu_baseline")
        with (args.run_dir / "single_gpu_same_initial.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-u", "-m", "distributed.mytorch_single_benchmark",
                 "--epochs", "10", "--batch-size", "16", "--learning-rate", "0.001",
                 "--device", "cuda", "--data-root", ".", "--train-list", "train.txt",
                 "--initial-parameters", "results/full_course/init_params.npy",
                 "--output", str(baseline)],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
            )

        aggregate_dir = args.run_dir / "aggregate"
        write_status(status, "aggregating")
        with (args.run_dir / "aggregation.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "tools/aggregate_multi_gpu_repeat.py",
                            "--run-dir", str(args.run_dir), "--baseline", str(baseline),
                            "--output-dir", str(aggregate_dir)], stdout=log, stderr=subprocess.STDOUT, check=True)
        with (args.run_dir / "plotting.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "tools/plot_multi_gpu_repeat.py",
                            "--aggregate-dir", str(aggregate_dir), "--output-dir", str(aggregate_dir)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

        archive = args.run_dir.parent / "multi_gpu_full_v1_results.tgz"
        write_status(status, "archiving")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(args.run_dir, arcname=args.run_dir.name)
        write_status(status, "complete", archive=str(archive))
        print(f"COMPLETE {archive}", flush=True)
    except Exception as error:
        write_status(status, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
