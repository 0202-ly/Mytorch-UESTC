import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np


def _argv_has(flag: str) -> bool:
    return flag in sys.argv


def _argv_value(flag: str):
    prefix = flag + "="
    for idx, arg in enumerate(sys.argv):
        if arg == flag:
            return sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _preconfigure_cupy_compiler():
    accelerators = _argv_value("--cupy-accelerators")
    if accelerators is None:
        os.environ.setdefault("CUPY_ACCELERATORS", "")
    elif accelerators.lower() in ("auto", "default"):
        pass
    elif accelerators.lower() in ("none", "off", "disabled"):
        os.environ["CUPY_ACCELERATORS"] = ""
    else:
        os.environ["CUPY_ACCELERATORS"] = accelerators

    if _argv_has("--allow-unsupported-nvcc"):
        flag = "-allow-unsupported-compiler"
        existing = os.environ.get("NVCC_PREPEND_FLAGS", "")
        if flag not in existing.split():
            os.environ["NVCC_PREPEND_FLAGS"] = f"{flag} {existing}".strip()


_preconfigure_cupy_compiler()

try:
    import cupy as cp
except ImportError:
    cp = None

from mytorch.dataloader import Dataloader
from mytorch.dataset import AutoDriveDataset
from mytorch.loss import MSELoss
from mytorch.optim import Adam
from mytorch.jit_train_rewrite import fuse_bn_relu_for_training
from model.resnet import ResNet18Original

try:
    import mytorch.jit as jit
except Exception:
    jit = None


@dataclass
class BenchConfig:
    data_root: str = "."
    train_list: str = "train.txt"
    val_list: str = "val.txt"
    results_dir: str = "results/donkeycar_resnet18_jit_benchmark"
    batch_size: int = 8
    epochs: int = 2
    lr: float = 1e-4
    max_train_batches: int = 50
    max_eval_batches: int = 50
    warmup_batches: int = 2
    seed: int = 2026
    output_dim: int = 1
    loss_log_interval: int = 20
    dump_jit_graph: bool = False
    eval_each_epoch: bool = False
    allow_unsupported_nvcc: bool = False
    cupy_accelerators: str = None
    include_jit_graph_reference: bool = False


def device_name():
    return "cuda" if cp is not None else "cpu"


def sync_device():
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def clear_cupy_pool():
    if cp is None:
        return
    sync_device()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def process_rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    proc = psutil.Process(os.getpid())
    return float(proc.memory_info().rss / (1024 ** 2))


def gpu_memory_used_mb():
    if cp is None:
        return None
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return float((total_bytes - free_bytes) / (1024 ** 2))
    except Exception:
        return None


def cupy_pool_used_mb():
    if cp is None:
        return None
    try:
        return float(cp.get_default_memory_pool().used_bytes() / (1024 ** 2))
    except Exception:
        return None


def stat_summary(values):
    if not values:
        return {"mean": None, "p50": None, "p90": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def set_seed(seed):
    np.random.seed(seed)
    if cp is not None:
        cp.random.seed(seed)


def to_float(value):
    if cp is not None and isinstance(value, cp.ndarray):
        value = value.get()
    elif hasattr(value, "data"):
        value = value.data
        if cp is not None and isinstance(value, cp.ndarray):
            value = value.get()
    return float(np.asarray(value).reshape(-1)[0])


def as_array(value):
    if hasattr(value, "data"):
        value = value.data
    if cp is not None and isinstance(value, cp.ndarray):
        return value.get()
    return np.asarray(value)


def batch_size_of(tensor):
    if hasattr(tensor, "shape") and callable(tensor.shape):
        return int(tensor.shape()[0])
    return int(tensor.data.shape[0])


def maybe_cuda(tensor, device):
    if device == "cuda":
        return tensor.cuda()
    return tensor


def resolve_list_path(data_root, list_path):
    if os.path.isabs(list_path):
        return list_path
    candidate = os.path.join(data_root, list_path)
    if os.path.exists(candidate):
        return candidate
    return list_path


def make_loader(mode, config, shuffle):
    list_path = config.train_list if mode == "train" else config.val_list
    dataset = AutoDriveDataset(mode=mode, data_root=config.data_root, list_path=list_path)
    return Dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=None,
    )


def check_runtime_requirements(config):
    missing = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python (required by AutoDriveDataset)")

    for path in (
        resolve_list_path(config.data_root, config.train_list),
        resolve_list_path(config.data_root, config.val_list),
    ):
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print("\nBenchmark cannot start because required inputs are missing:")
        for item in missing:
            print(f"  - {item}")
        print("\nInstall dependencies and/or pass --data-root to the DonkeyCar data directory.")
        return False

    return True


def build_model(config, use_train_fusion=False):
    set_seed(config.seed)
    model = ResNet18Original(num_classes=config.output_dim)
    if use_train_fusion:
        model = fuse_bn_relu_for_training(model, verbose=True)
    if device_name() == "cuda":
        model.cuda()
    return model


def timed_train_epoch(model, loader, optimizer, criterion, config):
    model.train()
    total_samples = 0
    total_batches = 0
    sampled_loss_sum = 0.0
    sampled_loss_count = 0
    batch_times_ms = []
    peak_process_rss = process_rss_mb()
    peak_gpu_used = gpu_memory_used_mb()
    peak_cupy_pool = cupy_pool_used_mb()

    sync_device()
    t0 = time.perf_counter()

    for batch_idx, (imgs, labels) in enumerate(loader, start=1):
        if config.max_train_batches and batch_idx > config.max_train_batches:
            break

        sync_device()
        batch_t0 = time.perf_counter()
        imgs = maybe_cuda(imgs, device_name())
        labels = maybe_cuda(labels, device_name())

        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        sync_device()
        batch_times_ms.append((time.perf_counter() - batch_t0) * 1000.0)

        total_samples += batch_size_of(imgs)
        total_batches += 1
        rss = process_rss_mb()
        gpu_used = gpu_memory_used_mb()
        pool_used = cupy_pool_used_mb()
        peak_process_rss = max([v for v in (peak_process_rss, rss) if v is not None], default=None)
        peak_gpu_used = max([v for v in (peak_gpu_used, gpu_used) if v is not None], default=None)
        peak_cupy_pool = max([v for v in (peak_cupy_pool, pool_used) if v is not None], default=None)

        if config.loss_log_interval and (
            batch_idx == 1 or batch_idx % config.loss_log_interval == 0
        ):
            sampled_loss_sum += to_float(loss)
            sampled_loss_count += 1

    sync_device()
    elapsed = time.perf_counter() - t0
    batch_stats = stat_summary(batch_times_ms)

    return {
        "samples": total_samples,
        "batches": total_batches,
        "time_sec": elapsed,
        "samples_per_sec": total_samples / elapsed if elapsed > 0 else 0.0,
        "batches_per_sec": total_batches / elapsed if elapsed > 0 else 0.0,
        "batch_ms_mean": batch_stats["mean"],
        "batch_ms_p50": batch_stats["p50"],
        "batch_ms_p90": batch_stats["p90"],
        "batch_ms_min": batch_stats["min"],
        "batch_ms_max": batch_stats["max"],
        "peak_process_rss_mb": peak_process_rss,
        "peak_gpu_used_mb": peak_gpu_used,
        "peak_cupy_pool_used_mb": peak_cupy_pool,
        "sampled_loss": (
            sampled_loss_sum / sampled_loss_count if sampled_loss_count else None
        ),
        "loss_samples": sampled_loss_count,
    }


def timed_forward_epoch(model_or_compiled, loader, config, run_loss=True):
    criterion = MSELoss()
    total_samples = 0
    total_batches = 0
    loss_sum = 0.0
    loss_count = 0
    batch_times_ms = []

    sync_device()
    t0 = time.perf_counter()

    for batch_idx, (imgs, labels) in enumerate(loader, start=1):
        if config.max_eval_batches and batch_idx > config.max_eval_batches:
            break

        sync_device()
        batch_t0 = time.perf_counter()
        imgs = maybe_cuda(imgs, device_name())
        labels = maybe_cuda(labels, device_name())

        pred = model_or_compiled(imgs)
        if run_loss:
            loss = criterion(pred, labels)
            if batch_idx == 1 or batch_idx % config.loss_log_interval == 0:
                loss_sum += to_float(loss)
                loss_count += 1

        total_samples += batch_size_of(imgs)
        total_batches += 1
        sync_device()
        batch_times_ms.append((time.perf_counter() - batch_t0) * 1000.0)

    sync_device()
    elapsed = time.perf_counter() - t0
    batch_stats = stat_summary(batch_times_ms)

    return {
        "samples": total_samples,
        "batches": total_batches,
        "time_sec": elapsed,
        "samples_per_sec": total_samples / elapsed if elapsed > 0 else 0.0,
        "batches_per_sec": total_batches / elapsed if elapsed > 0 else 0.0,
        "batch_ms_mean": batch_stats["mean"],
        "batch_ms_p50": batch_stats["p50"],
        "batch_ms_p90": batch_stats["p90"],
        "batch_ms_min": batch_stats["min"],
        "batch_ms_max": batch_stats["max"],
        "sampled_loss": loss_sum / loss_count if loss_count else None,
        "loss_samples": loss_count,
    }


def evaluate_regression(model_or_compiled, config):
    loader = make_loader("val", config, shuffle=False)
    criterion = MSELoss()
    if hasattr(model_or_compiled, "eval"):
        model_or_compiled.eval()

    total_samples = 0
    total_batches = 0
    sse = 0.0
    sae = 0.0
    acc05 = 0
    acc10 = 0
    batch_times_ms = []

    sync_device()
    t0 = time.perf_counter()
    for batch_idx, (imgs, labels) in enumerate(loader, start=1):
        if config.max_eval_batches and batch_idx > config.max_eval_batches:
            break

        sync_device()
        batch_t0 = time.perf_counter()
        imgs = maybe_cuda(imgs, device_name())
        labels = maybe_cuda(labels, device_name())
        pred = model_or_compiled(imgs)
        loss = criterion(pred, labels)
        pred_arr = as_array(pred).reshape(-1)
        label_arr = as_array(labels).reshape(-1)
        diff = pred_arr - label_arr
        abs_diff = np.abs(diff)
        sse += float(np.sum(diff * diff))
        sae += float(np.sum(abs_diff))
        acc05 += int(np.sum(abs_diff <= 0.05))
        acc10 += int(np.sum(abs_diff <= 0.10))
        total_samples += int(diff.size)
        total_batches += 1
        sync_device()
        batch_times_ms.append((time.perf_counter() - batch_t0) * 1000.0)

    sync_device()
    elapsed = time.perf_counter() - t0
    batch_stats = stat_summary(batch_times_ms)
    mse = sse / max(1, total_samples)
    if hasattr(model_or_compiled, "train"):
        model_or_compiled.train()
    return {
        "val_samples": total_samples,
        "val_batches": total_batches,
        "val_time_sec": elapsed,
        "val_mse": mse,
        "val_rmse": float(np.sqrt(mse)),
        "val_mae": sae / max(1, total_samples),
        "val_acc_at_0.05": acc05 / max(1, total_samples),
        "val_acc_at_0.10": acc10 / max(1, total_samples),
        "val_batch_ms_mean": batch_stats["mean"],
        "val_batch_ms_p50": batch_stats["p50"],
        "val_batch_ms_p90": batch_stats["p90"],
    }


def run_training_variant(
    run_name,
    config,
    use_train_fusion,
    use_jit_train=False,
    use_jit_experimental_conv_bn_fusion=False,
):
    print(f"\n=== True training run: {run_name} ===", flush=True)
    if use_jit_train and jit is None:
        print(f"[{run_name}] skipped: mytorch.jit import failed.", flush=True)
        return {
            "run_name": run_name,
            "phase": "train",
            "skipped": True,
            "skip_reason": "jit_import_failed",
        }, []

    train_loader = make_loader("train", config, shuffle=True)
    model = build_model(config, use_train_fusion=use_train_fusion)
    runner = (
        jit.compile_train(
            model,
            dump_graph=config.dump_jit_graph,
            experimental_conv_bn_fusion=use_jit_experimental_conv_bn_fusion,
        )
        if use_jit_train
        else model
    )
    optimizer = Adam(runner.parameters(), lr=config.lr)
    criterion = MSELoss()

    epoch_records = []
    wall_t0 = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        metrics = timed_train_epoch(runner, train_loader, optimizer, criterion, config)
        record = {
            "run_name": run_name,
            "phase": "train",
            "true_training": True,
            "jit_enabled": bool(use_jit_train),
            "train_fusion": bool(use_train_fusion),
            "jit_experimental_conv_bn_fusion": bool(use_jit_experimental_conv_bn_fusion),
            "training_forward_graph_executor": bool(use_jit_train),
            "compile_first_call_sec": None,
            "epoch": epoch,
            **metrics,
        }
        if config.eval_each_epoch:
            val_metrics = evaluate_regression(runner, config)
            record.update(val_metrics)
        else:
            val_metrics = None
        epoch_records.append(record)
        write_csv(os.path.join(config.results_dir, f"{run_name}_epoch_records_live.csv"), epoch_records)
        val_text = f" val_mse={val_metrics['val_mse']:.6f}" if val_metrics else ""
        print(
            f"[{run_name}] epoch {epoch:03d} "
            f"time={metrics['time_sec']:.3f}s "
            f"batch_p50={metrics['batch_ms_p50']:.3f}ms "
            f"throughput={metrics['samples_per_sec']:.2f} samples/s "
            f"loss~={metrics['sampled_loss']}"
            f"{val_text}",
            flush=True,
        )

    wall_time = time.perf_counter() - wall_t0
    final_val_metrics = evaluate_regression(runner, config)
    summary = summarize_records(run_name, epoch_records, wall_time)
    summary.update(final_val_metrics)
    return summary, epoch_records


def warmup_jit(compiled_model, config):
    warmup_loader = make_loader("train", config, shuffle=False)
    compile_time = None
    warmup_count = 0

    for batch_idx, (imgs, _) in enumerate(warmup_loader, start=1):
        if batch_idx > max(1, config.warmup_batches):
            break
        imgs = maybe_cuda(imgs, device_name())
        sync_device()
        t0 = time.perf_counter()
        compiled_model(imgs)
        sync_device()
        elapsed = time.perf_counter() - t0
        if compile_time is None:
            compile_time = elapsed
        warmup_count += 1

    return {
        "compile_first_call_sec": compile_time,
        "warmup_batches": warmup_count,
    }


def run_forward_variant(run_name, config, use_jit):
    print(f"\n=== Forward/eval run: {run_name} ===", flush=True)
    model = build_model(config, use_train_fusion=False)
    model.eval()

    if use_jit:
        if device_name() != "cuda":
            print(f"[{run_name}] skipped: JIT CUDA backend requires CuPy/CUDA.", flush=True)
            return {
                "run_name": run_name,
                "phase": "forward_eval",
                "skipped": True,
                "skip_reason": "cuda_unavailable",
            }, []
        if jit is None:
            print(f"[{run_name}] skipped: mytorch.jit import failed.", flush=True)
            return {
                "run_name": run_name,
                "phase": "forward_eval",
                "skipped": True,
                "skip_reason": "jit_import_failed",
            }, []

        compiled = jit.compile(model, dump_graph=config.dump_jit_graph)
        warmup = warmup_jit(compiled, config)
        runner = compiled
    else:
        warmup = {"compile_first_call_sec": None, "warmup_batches": 0}
        runner = model

    eval_loader = make_loader("train", config, shuffle=False)
    wall_t0 = time.perf_counter()
    metrics = timed_forward_epoch(runner, eval_loader, config, run_loss=True)
    wall_time = time.perf_counter() - wall_t0

    record = {
        "run_name": run_name,
        "phase": "forward_eval",
        "true_training": False,
        "jit_enabled": bool(use_jit),
        "train_fusion": False,
        "epoch": 1,
        **warmup,
        **metrics,
    }
    print(
        f"[{run_name}] time={metrics['time_sec']:.3f}s "
        f"throughput={metrics['samples_per_sec']:.2f} samples/s "
        f"compile_first_call={warmup['compile_first_call_sec']}"
    )
    return summarize_records(run_name, [record], wall_time), [record]


def summarize_records(run_name, records, wall_time):
    if not records:
        return {"run_name": run_name, "skipped": True}

    times = [r["time_sec"] for r in records if r.get("time_sec") is not None]
    throughputs = [
        r["samples_per_sec"] for r in records if r.get("samples_per_sec") is not None
    ]
    losses = [r.get("sampled_loss") for r in records if r.get("sampled_loss") is not None]
    batch_p50 = [r.get("batch_ms_p50") for r in records if r.get("batch_ms_p50") is not None]
    batch_mean = [r.get("batch_ms_mean") for r in records if r.get("batch_ms_mean") is not None]
    batch_p90 = [r.get("batch_ms_p90") for r in records if r.get("batch_ms_p90") is not None]
    gpu_peaks = [r.get("peak_gpu_used_mb") for r in records if r.get("peak_gpu_used_mb") is not None]
    pool_peaks = [r.get("peak_cupy_pool_used_mb") for r in records if r.get("peak_cupy_pool_used_mb") is not None]
    rss_peaks = [r.get("peak_process_rss_mb") for r in records if r.get("peak_process_rss_mb") is not None]

    return {
        "run_name": run_name,
        "phase": records[0].get("phase"),
        "true_training": records[0].get("true_training"),
        "jit_enabled": records[0].get("jit_enabled"),
        "train_fusion": records[0].get("train_fusion"),
        "jit_experimental_conv_bn_fusion": records[0].get("jit_experimental_conv_bn_fusion"),
        "epochs_or_passes": len(records),
        "total_wall_time_sec": wall_time,
        "avg_time_sec": float(np.mean(times)) if times else None,
        "avg_samples_per_sec": float(np.mean(throughputs)) if throughputs else None,
        "avg_batch_ms_p50": float(np.mean(batch_p50)) if batch_p50 else None,
        "avg_batch_ms_mean": float(np.mean(batch_mean)) if batch_mean else None,
        "avg_batch_ms_p90": float(np.mean(batch_p90)) if batch_p90 else None,
        "peak_gpu_used_mb": float(np.max(gpu_peaks)) if gpu_peaks else None,
        "peak_cupy_pool_used_mb": float(np.max(pool_peaks)) if pool_peaks else None,
        "peak_process_rss_mb": float(np.max(rss_peaks)) if rss_peaks else None,
        "last_sampled_loss": losses[-1] if losses else None,
        "compile_first_call_sec": records[0].get("compile_first_call_sec"),
    }


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def write_csv(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not records:
        return
    keys = sorted({k for record in records for k in record.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def add_speedups(summary):
    by_name = {item["run_name"]: item for item in summary["runs"]}

    train_base = by_name.get("train_eager_original", {})
    train_fused = by_name.get("train_eager_dynamic_fused", {})
    train_forward_graph = by_name.get("train_forward_graph_executor", {})
    train_jit_fused = by_name.get("train_jit_fused", {})
    forward_base = by_name.get("forward_eager_eval", {})
    forward_jit = by_name.get("forward_jit_eval", {})

    def ratio(a, b):
        if not a or not b:
            return None
        return a / b if b else None

    summary["speedups"] = {
        "true_training_dynamic_fusion_time_speedup": ratio(
            train_base.get("avg_time_sec"),
            train_fused.get("avg_time_sec"),
        ),
        "true_training_dynamic_fusion_throughput_gain": ratio(
            train_fused.get("avg_samples_per_sec"),
            train_base.get("avg_samples_per_sec"),
        ),
        "training_forward_graph_time_speedup": ratio(
            train_base.get("avg_time_sec"),
            train_forward_graph.get("avg_time_sec"),
        ),
        "training_forward_graph_throughput_gain": ratio(
            train_forward_graph.get("avg_samples_per_sec"),
            train_base.get("avg_samples_per_sec"),
        ),
        "jit_fused_training_time_speedup": ratio(
            train_base.get("avg_time_sec"),
            train_jit_fused.get("avg_time_sec"),
        ),
        "jit_fused_training_throughput_gain": ratio(
            train_jit_fused.get("avg_samples_per_sec"),
            train_base.get("avg_samples_per_sec"),
        ),
        "jit_forward_time_speedup": ratio(
            forward_base.get("avg_time_sec"),
            forward_jit.get("avg_time_sec"),
        ),
        "jit_forward_throughput_gain": ratio(
            forward_jit.get("avg_samples_per_sec"),
            forward_base.get("avg_samples_per_sec"),
        ),
    }


def write_training_ablation_outputs(config, run_summaries):
    selected_names = [
        "train_eager_original",
        "train_eager_dynamic_fused",
        "train_jit_fused",
    ]
    labels = {
        "train_eager_original": "MyTorch no fusion",
        "train_eager_dynamic_fused": "Dynamic fused training",
        "train_jit_fused": "JIT fused training",
    }
    rows = []
    by_name = {row.get("run_name"): row for row in run_summaries}
    base = by_name.get("train_eager_original", {})
    for name in selected_names:
        row = by_name.get(name)
        if not row:
            continue
        avg_time = row.get("avg_time_sec")
        base_time = base.get("avg_time_sec")
        rows.append({
            "variant": labels[name],
            "run_name": name,
            "epochs": row.get("epochs_or_passes"),
            "avg_epoch_time_sec": avg_time,
            "total_wall_time_sec": row.get("total_wall_time_sec"),
            "samples_per_sec": row.get("avg_samples_per_sec"),
            "batch_p50_ms": row.get("avg_batch_ms_p50"),
            "batch_p90_ms": row.get("avg_batch_ms_p90"),
            "peak_gpu_used_mb": row.get("peak_gpu_used_mb"),
            "peak_cupy_pool_used_mb": row.get("peak_cupy_pool_used_mb"),
            "peak_process_rss_mb": row.get("peak_process_rss_mb"),
            "val_mse": row.get("val_mse"),
            "val_rmse": row.get("val_rmse"),
            "val_mae": row.get("val_mae"),
            "val_acc_at_0.10": row.get("val_acc_at_0.10"),
            "time_speedup_vs_no_fusion": (base_time / avg_time) if base_time and avg_time else None,
        })

    csv_path = os.path.join(config.results_dir, "training_fusion_ablation.csv")
    md_path = os.path.join(config.results_dir, "training_fusion_ablation.md")
    write_csv(csv_path, rows)

    def fmt(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# MyTorch Training Fusion / JIT Ablation",
        "",
        "Compares MyTorch eager training without fusion, dynamic fused training, and JIT fused training.",
        "",
        f"- Data root: `{config.data_root}`",
        f"- Train list: `{config.train_list}`",
        f"- Val list: `{config.val_list}`",
        f"- Epochs: {config.epochs}",
        f"- Batch size: {config.batch_size}",
        f"- Max train batches per epoch: {config.max_train_batches or 'all'}",
        f"- Max eval batches: {config.max_eval_batches or 'all'}",
        f"- Device: {device_name()}",
        "",
        "| variant | epoch time s | batch p50 ms | batch p90 ms | samples/s | peak GPU MB | CuPy pool MB | val MSE | val MAE | acc@0.10 | speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["variant"],
                fmt(row["avg_epoch_time_sec"]),
                fmt(row["batch_p50_ms"]),
                fmt(row["batch_p90_ms"]),
                fmt(row["samples_per_sec"]),
                fmt(row["peak_gpu_used_mb"]),
                fmt(row["peak_cupy_pool_used_mb"]),
                fmt(row["val_mse"]),
                fmt(row["val_mae"]),
                fmt(row["val_acc_at_0.10"]),
                fmt(row["time_speedup_vs_no_fusion"]),
            ])
            + " |"
        )
    reference = by_name.get("train_forward_graph_executor")
    if reference:
        lines.extend([
            "",
            "## JIT Graph Reference",
            "",
            "This optional diagnostic run uses the JIT training graph executor without experimental Conv-BN fusion. "
            "It is not part of the main three-way ablation, but helps separate graph execution overhead from fusion gains.",
            "",
            "| run | epoch time s | batch p50 ms | samples/s | peak GPU MB | val MSE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            "| "
            + " | ".join([
                "JIT graph executor without Conv-BN fusion",
                fmt(reference.get("avg_time_sec")),
                fmt(reference.get("avg_batch_ms_p50")),
                fmt(reference.get("avg_samples_per_sec")),
                fmt(reference.get("peak_gpu_used_mb")),
                fmt(reference.get("val_mse")),
            ])
            + " |",
        ])
    lines.extend([
        "",
        "Notes:",
        "- `batch p50/p90` are measured inside the true training loop with device synchronization.",
        "- `val MSE/MAE/acc@0.10` are evaluated once after each variant finishes training.",
        "- `JIT fused training` first applies the same BN/ReLU training rewrite as `Dynamic fused training`, then uses `jit.compile_train(..., experimental_conv_bn_fusion=False)` for graph execution. Experimental Conv-BN fusion is not part of the main ablation.",
        "",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return csv_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ResNet18 on DonkeyCar: eager training, training forward graph execution, "
            "and JIT optimized forward/eval performance."
        )
    )
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--val-list", default="val.txt")
    parser.add_argument("--results-dir", default="results/donkeycar_resnet18_jit_benchmark")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int, default=50)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--loss-log-interval", type=int, default=20)
    parser.add_argument("--dump-jit-graph", action="store_true")
    parser.add_argument(
        "--cupy-accelerators",
        default=None,
        help="none/off/disabled disables CuPy accelerators; auto leaves CuPy defaults.",
    )
    parser.add_argument("--allow-unsupported-nvcc", action="store_true")
    parser.add_argument(
        "--eval-each-epoch",
        action="store_true",
        help="Also compute validation metrics after each epoch. Default evaluates once per variant to keep timing cleaner.",
    )
    parser.add_argument(
        "--include-jit-graph-reference",
        action="store_true",
        help="Also run the optional JIT training graph executor without experimental Conv-BN fusion.",
    )
    parser.add_argument(
        "--include-experimental-conv-bn-jit",
        action="store_true",
        help="Also run the optional experimental JIT Conv-BN fusion diagnostic. This is not part of the main three-way ablation.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only run eager/JIT forward-eval benchmark.",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Only run true training benchmark.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchConfig(
        data_root=args.data_root,
        train_list=args.train_list,
        val_list=args.val_list,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        warmup_batches=args.warmup_batches,
        seed=args.seed,
        output_dim=args.output_dim,
        loss_log_interval=args.loss_log_interval,
        dump_jit_graph=args.dump_jit_graph,
        eval_each_epoch=args.eval_each_epoch,
        allow_unsupported_nvcc=args.allow_unsupported_nvcc,
        cupy_accelerators=args.cupy_accelerators,
        include_jit_graph_reference=args.include_jit_graph_reference,
    )

    os.makedirs(config.results_dir, exist_ok=True)

    print("Config:", flush=True)
    print(json.dumps(asdict(config), indent=2, ensure_ascii=False), flush=True)
    print(f"Device: {device_name()}", flush=True)
    print(
        "Note: the training forward graph executor traces and optimizes only "
        "the forward graph; loss.backward() still uses eager autograd and is "
        "included in the timing.",
        flush=True,
    )

    if not check_runtime_requirements(config):
        return

    all_records = []
    run_summaries = []

    if not args.skip_training:
        clear_cupy_pool()
        summary, records = run_training_variant(
            "train_eager_original",
            config,
            use_train_fusion=False,
            use_jit_train=False,
        )
        run_summaries.append(summary)
        all_records.extend(records)

        clear_cupy_pool()
        summary, records = run_training_variant(
            "train_eager_dynamic_fused",
            config,
            use_train_fusion=True,
            use_jit_train=False,
        )
        run_summaries.append(summary)
        all_records.extend(records)

        if args.include_jit_graph_reference:
            clear_cupy_pool()
            summary, records = run_training_variant(
                "train_forward_graph_executor",
                config,
                use_train_fusion=False,
                use_jit_train=True,
                use_jit_experimental_conv_bn_fusion=False,
            )
            run_summaries.append(summary)
            all_records.extend(records)

        clear_cupy_pool()
        summary, records = run_training_variant(
            "train_jit_fused",
            config,
            use_train_fusion=True,
            use_jit_train=True,
            use_jit_experimental_conv_bn_fusion=False,
        )
        run_summaries.append(summary)
        all_records.extend(records)

        if args.include_experimental_conv_bn_jit:
            clear_cupy_pool()
            summary, records = run_training_variant(
                "train_jit_experimental_conv_bn_fused",
                config,
                use_train_fusion=False,
                use_jit_train=True,
                use_jit_experimental_conv_bn_fusion=True,
            )
            run_summaries.append(summary)
            all_records.extend(records)

    if not args.skip_forward:
        clear_cupy_pool()
        summary, records = run_forward_variant(
            "forward_eager_eval",
            config,
            use_jit=False,
        )
        run_summaries.append(summary)
        all_records.extend(records)

        clear_cupy_pool()
        summary, records = run_forward_variant(
            "forward_jit_eval",
            config,
            use_jit=True,
        )
        run_summaries.append(summary)
        all_records.extend(records)

    final_summary = {
        "config": asdict(config),
        "device": device_name(),
        "notes": {
            "jit_scope": (
                "The training path is a forward graph optimizer/executor: it returns "
                "tensors with autograd creators and leaves backward to eager autograd."
            ),
            "training_scope": (
                "True training benchmark compares original eager autograd, "
                "manual training rewrite fusion, and JIT graph execution on the same "
                "training-rewritten model."
            ),
        },
        "runs": run_summaries,
    }
    add_speedups(final_summary)

    summary_path = os.path.join(config.results_dir, "benchmark_summary.json")
    records_path = os.path.join(config.results_dir, "epoch_records.csv")
    write_json(summary_path, final_summary)
    write_csv(records_path, all_records)
    ablation_csv, ablation_md = write_training_ablation_outputs(config, run_summaries)

    print("\nSummary:", flush=True)
    print(json.dumps(final_summary["speedups"], indent=2, ensure_ascii=False), flush=True)
    print("\nSaved:", flush=True)
    print(f"  {summary_path}", flush=True)
    print(f"  {records_path}", flush=True)
    print(f"  {ablation_csv}", flush=True)
    print(f"  {ablation_md}", flush=True)


if __name__ == "__main__":
    main()
