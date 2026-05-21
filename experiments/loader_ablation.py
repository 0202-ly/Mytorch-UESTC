import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_donkeycar_resnet18_jit_vs_pytorch as base
from train_donkeycar_resnet18_jit_vs_pytorch import (
    ExperimentConfig,
    estimate_resnet18_flops,
    run_mytorch_jit,
    write_json,
)


class GpuUtilMonitor:
    """Small nvidia-smi based monitor for PPT-level GPU utilization evidence."""

    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = float(interval_sec)
        self.samples: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Optional[float]]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {"gpu_util_avg_percent": None, "gpu_util_max_percent": None}
        arr = np.asarray(self.samples, dtype=np.float64)
        return {
            "gpu_util_avg_percent": float(np.mean(arr)),
            "gpu_util_max_percent": float(np.max(arr)),
        }

    def _loop(self) -> None:
        while self._running:
            value = self._query_once()
            if value is not None:
                self.samples.append(value)
            time.sleep(self.interval_sec)

    @staticmethod
    def _query_once() -> Optional[float]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            )
            if proc.returncode != 0:
                return None
            first = proc.stdout.strip().splitlines()[0].strip()
            return float(first)
        except Exception:
            return None


def none_if_zero(value: int) -> Optional[int]:
    return None if value == 0 else value


def safe_get(mapping: Dict[str, Any], *keys: str):
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def ratio(a, b):
    if a is None or b in (None, 0):
        return None
    return float(a) / float(b)


def improvement(baseline, current):
    if baseline in (None, 0) or current is None:
        return None
    return (float(baseline) - float(current)) / float(baseline)


def make_config(args: argparse.Namespace, output_dir: str, loader: str) -> ExperimentConfig:
    device = "cpu" if args.cpu else ("cuda" if base.cp is not None else "cpu")
    return ExperimentConfig(
        data_root=args.data_root,
        train_list=args.train_list,
        val_list=args.val_list,
        results_dir=output_dir,
        backends=["mytorch_jit"],
        order=["mytorch_jit"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dim=1,
        image_height=args.image_height,
        image_width=args.image_width,
        seed=args.seed,
        device=device,
        max_train_batches=none_if_zero(args.max_train_batches),
        max_val_batches=none_if_zero(args.max_val_batches),
        loss_log_interval=args.loss_log_interval,
        resource_interval_sec=args.resource_interval_sec,
        torch_num_workers=0,
        torch_prefetch_factor=2,
        torch_persistent_workers=False,
        torch_cudnn_benchmark=False,
        mytorch_loader=loader,
        mytorch_num_workers=args.workers,
        mytorch_prefetch_factor=args.prefetch_factor,
        model_parallel="none",
        jit_profile=False,
        jit_dump_graph=False,
        jit_experimental_conv_bn_fusion=args.jit_experimental_conv_bn_fusion,
        allow_unsupported_nvcc=args.allow_unsupported_nvcc,
        cupy_accelerators=args.cupy_accelerators,
    )


class MyTorchSeededSyncDataLoader:
    """Synchronous loader with the same seeded batch order as the async loader."""

    def __init__(self, dataset, batch_size: int, shuffle: bool, seed: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        self._batches = []
        self._pos = 0
        self._stats = {
            "loader": "mytorch_seeded_sync",
            "num_workers": 0,
            "prefetch_factor": 0,
            "produced_batches": 0,
            "consumed_batches": 0,
            "queue_size": 0,
        }

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _collate(self, samples):
        xs, ys = zip(*samples)
        x = base.Tensor(np.stack([item.data for item in xs], axis=0))
        y = base.Tensor(np.stack([item.data for item in ys], axis=0))
        return x, y

    def __iter__(self):
        self._epoch += 1
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(indices)
        self._batches = [
            indices[start:start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        self._pos = 0
        self._stats["produced_batches"] = len(self._batches)
        self._stats["consumed_batches"] = 0
        return self

    def __next__(self):
        if self._pos >= len(self._batches):
            raise StopIteration
        batch_indices = self._batches[self._pos]
        self._pos += 1
        self._stats["consumed_batches"] += 1
        return self._collate([self.dataset[int(idx)] for idx in batch_indices])

    def get_stats(self):
        return dict(self._stats)


def install_loader_factory(shuffle_train: bool) -> None:
    def make_mytorch_loaders(config: ExperimentConfig):
        train_ds = base.MyTorchDonkeyCarDataset(
            config.data_root,
            "train",
            config.image_height,
            config.image_width,
            config.train_list,
        )
        val_ds = base.MyTorchDonkeyCarDataset(
            config.data_root,
            "val",
            config.image_height,
            config.image_width,
            config.val_list,
        )
        if config.mytorch_loader == "async":
            train_loader = base.MyTorchThreadedDataLoader(
                train_ds,
                batch_size=config.batch_size,
                shuffle=shuffle_train,
                num_workers=config.mytorch_num_workers,
                prefetch_factor=config.mytorch_prefetch_factor,
                seed=config.seed,
            )
            val_loader = base.MyTorchThreadedDataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.mytorch_num_workers,
                prefetch_factor=config.mytorch_prefetch_factor,
                seed=config.seed + 9973,
            )
        else:
            train_loader = MyTorchSeededSyncDataLoader(
                train_ds,
                batch_size=config.batch_size,
                shuffle=shuffle_train,
                seed=config.seed,
            )
            val_loader = MyTorchSeededSyncDataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                seed=config.seed + 9973,
            )
        return train_loader, val_loader, len(train_ds), len(val_ds)

    base.make_mytorch_loaders = make_mytorch_loaders


def summarize_run(loader: str, result: Dict[str, Any], gpu_util: Dict[str, Optional[float]]) -> Dict[str, Any]:
    batch = result.get("batch_summary", {})
    final_val = result.get("final_val", {})
    resources = result.get("resources", {})
    epochs = len(result.get("history", [])) or None
    total_train_time = result.get("total_train_time_sec")
    avg_epoch_time = ratio(total_train_time, epochs) if epochs else None
    row = {
        "loader": loader,
        "epochs": epochs,
        "train_samples": result.get("train_size"),
        "val_samples": result.get("val_size"),
        "avg_epoch_time_sec": avg_epoch_time,
        "total_train_time_sec": total_train_time,
        "wall_time_sec": result.get("wall_time_sec"),
        "samples_per_sec": result.get("samples_per_sec"),
        "end_to_end_samples_per_sec": result.get("end_to_end_samples_per_sec"),
        "loader_fetch_p50_ms": safe_get(batch, "loader_fetch_ms", "p50"),
        "loader_fetch_mean_ms": safe_get(batch, "loader_fetch_ms", "mean"),
        "loader_fetch_p90_ms": safe_get(batch, "loader_fetch_ms", "p90"),
        "queue_wait_p50_ms": safe_get(batch, "loader_fetch_ms", "p50"),
        "total_batch_p50_ms": safe_get(batch, "total_batch_ms", "p50"),
        "total_batch_p90_ms": safe_get(batch, "total_batch_ms", "p90"),
        "forward_loss_p50_ms": safe_get(batch, "forward_loss_ms", "p50"),
        "backward_p50_ms": safe_get(batch, "backward_ms", "p50"),
        "optimizer_step_p50_ms": safe_get(batch, "optimizer_step_ms", "p50"),
        "val_mse": final_val.get("mse"),
        "val_rmse": final_val.get("rmse"),
        "val_mae": final_val.get("mae"),
        "val_samples_per_sec": final_val.get("samples_per_sec"),
        "gpu_used_max_mb": resources.get("max_gpu_used_mb"),
        "cupy_pool_max_mb": resources.get("max_cupy_pool_used_mb"),
        "process_rss_max_mb": resources.get("max_process_rss_mb"),
        "process_cpu_avg_percent": resources.get("avg_process_cpu_percent"),
        "gpu_util_avg_percent": gpu_util.get("gpu_util_avg_percent"),
        "gpu_util_max_percent": gpu_util.get("gpu_util_max_percent"),
    }
    stats = result.get("data_loader_stats") or {}
    for key in ["produced_batches", "consumed_batches", "queue_size", "num_workers", "prefetch_factor"]:
        row[f"loader_{key}"] = stats.get(key)
    return row


def add_relative_metrics(rows: List[Dict[str, Any]]) -> None:
    by_loader = {row["loader"]: row for row in rows}
    sync = by_loader.get("sync")
    if not sync:
        return
    for row in rows:
        row["speedup_vs_sync"] = ratio(sync.get("avg_epoch_time_sec"), row.get("avg_epoch_time_sec"))
        row["throughput_gain_vs_sync"] = ratio(row.get("samples_per_sec"), sync.get("samples_per_sec"))
        row["loader_fetch_reduction_vs_sync"] = improvement(
            sync.get("loader_fetch_p50_ms"),
            row.get("loader_fetch_p50_ms"),
        )
        row["batch_p50_reduction_vs_sync"] = improvement(
            sync.get("total_batch_p50_ms"),
            row.get("total_batch_p50_ms"),
        )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# DonkeyCar MyTorch Loader Ablation",
        "",
        "Compares synchronous MyTorch dataloader against producer-consumer async loader with the same model/training configuration.",
        "",
        f"- Train list: `{args.train_list}`",
        f"- Val list: `{args.val_list}`",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Max train batches per epoch: {args.max_train_batches or 'all'}",
        f"- Max val batches: {args.max_val_batches or 'all'}",
        f"- Train shuffle: {bool(args.shuffle_train)}",
        f"- Async workers: {args.workers}",
        f"- Async prefetch factor: {args.prefetch_factor}",
        "",
        "| loader | epoch time s | samples/s | loader fetch p50 ms | batch p50 ms | GPU util avg % | peak GPU MB | val MSE | speedup | fetch reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["loader"],
                    fmt(row.get("avg_epoch_time_sec")),
                    fmt(row.get("samples_per_sec")),
                    fmt(row.get("loader_fetch_p50_ms")),
                    fmt(row.get("total_batch_p50_ms")),
                    fmt(row.get("gpu_util_avg_percent")),
                    fmt(row.get("gpu_used_max_mb")),
                    fmt(row.get("val_mse")),
                    fmt(row.get("speedup_vs_sync")),
                    fmt(row.get("loader_fetch_reduction_vs_sync")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
        "Notes:",
        "- `loader fetch p50` is the consumer-side wait time for the next batch. For async mode it approximates queue wait after producer prefetching.",
        "- GPU utilization is sampled with `nvidia-smi` when available; otherwise the column is `N/A`.",
        "- Validation MSE is a sanity metric; the primary target of this ablation is data path latency and throughput.",
        "- If `max_train_batches` is smaller than a full epoch, validation MSE is not suitable for model-quality conclusions because BatchNorm running statistics are under-trained.",
        "",
    ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync vs async MyTorch dataloader ablation for DonkeyCar ResNet18 JIT.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default=os.path.join("splits", "temporal_block_gap20", "train.txt"))
    parser.add_argument("--val-list", default=os.path.join("splits", "temporal_block_gap20", "val.txt"))
    parser.add_argument("--results-dir", default=os.path.join("results", "loader_ablation"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-batches", type=int, default=50, help="0 means all train batches.")
    parser.add_argument("--max-val-batches", type=int, default=50, help="0 means all validation batches.")
    parser.add_argument("--loss-log-interval", type=int, default=20)
    parser.add_argument("--resource-interval-sec", type=float, default=0.5)
    parser.add_argument("--gpu-util-interval-sec", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--shuffle-train", dest="shuffle_train", action="store_true", help="Enable train shuffle.")
    parser.add_argument(
        "--no-shuffle-train",
        dest="shuffle_train",
        action="store_false",
        help="Disable train shuffle only for diagnosing temporal-order effects.",
    )
    parser.set_defaults(shuffle_train=True)
    parser.add_argument(
        "--jit-experimental-conv-bn-fusion",
        dest="jit_experimental_conv_bn_fusion",
        action="store_true",
        help="Enable the same experimental Conv-BN fusion used by the JIT fused benchmark.",
    )
    parser.add_argument(
        "--no-jit-experimental-conv-bn-fusion",
        dest="jit_experimental_conv_bn_fusion",
        action="store_false",
        help="Disable experimental Conv-BN fusion and run the default BN-only training JIT path.",
    )
    parser.set_defaults(jit_experimental_conv_bn_fusion=True)
    parser.add_argument("--cupy-accelerators", default=None)
    parser.add_argument("--allow-unsupported-nvcc", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.results_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    flops = estimate_resnet18_flops(args.image_height, args.image_width, 1)
    write_json(str(output_dir / "config.json"), vars(args))
    write_json(str(output_dir / "flops_estimate.json"), flops)
    install_loader_factory(shuffle_train=bool(args.shuffle_train))

    rows: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}
    for loader in ["sync", "async"]:
        print("\n" + "=" * 88)
        print(f"Running MyTorch loader ablation: {loader}")
        print("=" * 88)
        config = make_config(args, str(output_dir), loader)
        variant_dir = output_dir / loader
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_json(str(variant_dir / "config.json"), asdict(config))

        monitor = GpuUtilMonitor(args.gpu_util_interval_sec)
        monitor.start()
        result = run_mytorch_jit(config, str(variant_dir), flops)
        gpu_util = monitor.stop()
        result["gpu_utilization"] = gpu_util
        write_json(str(variant_dir / "result.json"), result)

        rows.append(summarize_run(loader, result, gpu_util))
        results[loader] = result

    add_relative_metrics(rows)
    summary = {
        "config": vars(args),
        "output_dir": str(output_dir),
        "rows": rows,
        "results": results,
    }
    write_json(str(output_dir / "summary.json"), summary)
    write_csv(output_dir / "summary.csv", rows)
    write_markdown(output_dir / "summary.md", args, rows)

    print("\nSaved loader ablation results:")
    print(output_dir / "summary.md")
    print(output_dir / "summary.csv")


if __name__ == "__main__":
    main()
