"""Profile real MyTorch DonkeyCar steps and GPU/CPU resource use."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import cupy as cp
import numpy as np

from model.resnet import ResNet18Original
from mytorch.dataset import AutoDriveDataset
from mytorch.dataloader import Dataloader
from mytorch.loss import MSELoss
from distributed.mytorch_adapter import flatten_parameters, flatten_gradients, load_parameters


class IndexedDataset:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def cpu_times() -> tuple[int, int, float]:
    with Path("/proc/stat").open(encoding="utf-8") as handle:
        fields = [int(value) for value in handle.readline().split()[1:]]
    process = os.times()
    return sum(fields), fields[3] + fields[4], process.user + process.system


def sample_resources(out: list[dict], stop: threading.Event, gpu_index: int) -> None:
    previous = None
    while not stop.is_set():
        timestamp = time.time()
        current = (*cpu_times(), time.monotonic())
        try:
            raw = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu_index),
                 "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], text=True,
            ).strip()
            values = [float(value.strip()) for value in raw.split(",")]
            if previous is not None:
                delta_total = current[0] - previous[0]
                delta_idle = current[1] - previous[1]
                delta_process = current[2] - previous[2]
                delta_wall = current[3] - previous[3]
                out.append({
                    "timestamp": timestamp,
                    "gpu_index": gpu_index,
                    "gpu_util_pct": values[0],
                    "memory_used_mib": values[1],
                    "memory_total_mib": values[2],
                    "temperature_c": values[3],
                    "power_w": values[4],
                    "cpu_system_util_pct": 100 * (delta_total - delta_idle) / delta_total if delta_total else 0,
                    "cpu_process_util_pct_one_core": 100 * delta_process / delta_wall if delta_wall else 0,
                })
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
        previous = current
        stop.wait(0.5)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no samples collected for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--initial-parameters", required=True)
    parser.add_argument("--output-dir", default="results/mytorch_gpu_profile_v2")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="Physical GPU ID for nvidia-smi; set CUDA_VISIBLE_DEVICES to the same ID")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = ResNet18Original(num_classes=1).train()
    model.cuda()
    load_parameters(model, np.asarray(np.load(args.initial_parameters), dtype=np.float32))
    dataset = AutoDriveDataset("train", data_root=args.data_root, list_path=args.train_list)
    loader = Dataloader(IndexedDataset(dataset, np.arange(len(dataset))), batch_size=args.batch_size, shuffle=True)
    criterion = MSELoss(fused=False)
    rows: list[dict] = []
    resources: list[dict] = []
    stop = threading.Event()
    monitor = threading.Thread(target=sample_resources, args=(resources, stop, args.gpu_index), daemon=True)
    monitor.start()
    try:
        iterator = iter(loader)
        for step in range(args.batches):
            start = time.perf_counter()
            try:
                x, y = next(iterator)
            except StopIteration:
                break
            load_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            x = x.cuda()
            y = y.cuda()
            cp.cuda.Stream.null.synchronize()
            transfer_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            model.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            cp.cuda.Stream.null.synchronize()
            forward_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            loss.backward()
            cp.cuda.Stream.null.synchronize()
            backward_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            params = flatten_parameters(model)
            grad = flatten_gradients(model)
            load_parameters(model, params - args.lr * grad)
            cp.cuda.Stream.null.synchronize()
            update_ms = (time.perf_counter() - start) * 1000

            total_ms = load_ms + transfer_ms + forward_ms + backward_ms + update_ms
            samples = int(x.shape()[0])
            rows.append({
                "step": step,
                "data_load_ms": load_ms,
                "data_transfer_ms": transfer_ms,
                "data_prepare_ms": load_ms + transfer_ms,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_update_ms": update_ms,
                "step_ms": total_ms,
                "samples": samples,
                "samples_per_sec": samples / (total_ms / 1000),
                "loss": float(loss.data.get() if hasattr(loss.data, "get") else loss.data),
            })
    finally:
        stop.set()
        monitor.join(timeout=2)

    write_csv(out / "step_metrics.csv", rows)
    write_csv(out / "gpu_cpu_resources.csv", resources)
    summary = {
        "steps": len(rows),
        "samples": sum(row["samples"] for row in rows),
        "avg_step_ms": float(np.mean([row["step_ms"] for row in rows])),
        "avg_data_load_ms": float(np.mean([row["data_load_ms"] for row in rows])),
        "avg_data_transfer_ms": float(np.mean([row["data_transfer_ms"] for row in rows])),
        "avg_forward_ms": float(np.mean([row["forward_ms"] for row in rows])),
        "avg_backward_ms": float(np.mean([row["backward_ms"] for row in rows])),
        "avg_optimizer_update_ms": float(np.mean([row["optimizer_update_ms"] for row in rows])),
        "avg_samples_per_sec": float(np.mean([row["samples_per_sec"] for row in rows])),
        "gpu_util_avg_pct": float(np.mean([row["gpu_util_pct"] for row in resources])),
        "gpu_util_max_pct": float(np.max([row["gpu_util_pct"] for row in resources])),
        "memory_peak_mib": float(np.max([row["memory_used_mib"] for row in resources])),
        "temperature_max_c": float(np.max([row["temperature_c"] for row in resources])),
        "power_max_w": float(np.max([row["power_w"] for row in resources])),
        "cpu_system_avg_pct": float(np.mean([row["cpu_system_util_pct"] for row in resources])),
        "cpu_process_avg_pct_one_core": float(np.mean([row["cpu_process_util_pct_one_core"] for row in resources])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
