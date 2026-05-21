import argparse
import json
import os
import sys
import time

import numpy as np


def _argv_value(flag):
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
        # Avoid CuPy's CUB reduction path by default. On Windows that path can
        # invoke NVCC for reductions such as argmax before the benchmark reaches
        # the JIT variants.
        os.environ.setdefault("CUPY_ACCELERATORS", "")
    elif accelerators.lower() in ("auto", "default"):
        pass
    elif accelerators.lower() in ("none", "off", "disabled"):
        os.environ["CUPY_ACCELERATORS"] = ""
    else:
        os.environ["CUPY_ACCELERATORS"] = accelerators

    if "--allow-unsupported-nvcc" in sys.argv:
        flag = "-allow-unsupported-compiler"
        existing = os.environ.get("NVCC_PREPEND_FLAGS", "")
        if flag not in existing.split():
            os.environ["NVCC_PREPEND_FLAGS"] = f"{flag} {existing}".strip()


_preconfigure_cupy_compiler()

try:
    import cupy as cp
except ImportError:
    cp = None

import mytorch.jit as jit
from model.resnet import ResNet18Original
from mytorch.function import MSE
from mytorch.jit_train_rewrite import fuse_bn_relu_for_training
from mytorch.optim import SGD
from mytorch.tensor import Tensor


VALID_VARIANTS = (
    "eager_original",
    "eager_dynamic_fused",
    "jit_training_backend",
    "jit_training_bn_only",
    "jit_training_conv_bn_experimental",
)
DEFAULT_VARIANTS = (
    "eager_original",
    "eager_dynamic_fused",
    "jit_training_backend",
    "jit_training_conv_bn_experimental",
)


def sync_device():
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def clear_cupy_pool():
    if cp is None:
        return
    sync_device()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def set_seed(seed):
    np.random.seed(seed)
    if cp is not None:
        cp.random.seed(seed)


def _optional_pool_metric(pool, method_name):
    method = getattr(pool, method_name, None)
    if method is None:
        return None
    try:
        return int(method())
    except (AttributeError, TypeError, RuntimeError):
        return None


def memory_snapshot():
    if cp is None:
        return {
            "cupy_pool_used_bytes": None,
            "cupy_pool_total_bytes": None,
            "cupy_pinned_used_bytes": None,
            "cupy_pinned_total_bytes": None,
            "gpu_free_bytes": None,
            "gpu_total_bytes": None,
        }

    pool = cp.get_default_memory_pool()
    pinned = cp.get_default_pinned_memory_pool()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return {
        "cupy_pool_used_bytes": _optional_pool_metric(pool, "used_bytes"),
        "cupy_pool_total_bytes": _optional_pool_metric(pool, "total_bytes"),
        "cupy_pinned_used_bytes": _optional_pool_metric(pinned, "used_bytes"),
        "cupy_pinned_total_bytes": _optional_pool_metric(pinned, "total_bytes"),
        "gpu_free_bytes": int(free_bytes),
        "gpu_total_bytes": int(total_bytes),
    }


def max_memory(a, b):
    out = {}
    for key in a:
        av = a[key]
        bv = b[key]
        out[key] = None if av is None or bv is None else max(av, bv)
    return out


def make_batch(args, device):
    set_seed(args.data_seed)
    x_np = np.random.randn(
        args.batch_size,
        3,
        args.image_height,
        args.image_width,
    ).astype(np.float32)
    y_np = np.random.randn(args.batch_size, args.output_dim).astype(np.float32)

    x = Tensor(x_np, requires_grad=False)
    y = Tensor(y_np, requires_grad=False)
    if device == "cuda":
        x.cuda()
        y.cuda()
    return x, y


def build_model(args, variant, device):
    set_seed(args.model_seed)
    model = ResNet18Original(num_classes=args.output_dim).train()

    if variant == "eager_dynamic_fused":
        model = fuse_bn_relu_for_training(model, verbose=False)

    if device == "cuda":
        model.cuda()

    if variant in {
        "jit_training_backend",
        "jit_training_bn_only",
        "jit_training_conv_bn_experimental",
    }:
        disable_conv_bn_fusion = True if args.jit_disable_conv_bn_fusion else None
        if variant == "jit_training_bn_only":
            disable_conv_bn_fusion = True
        return jit.compile(
            model,
            training=True,
            dump_graph=args.dump_jit_graph,
            profile=args.jit_profile,
            disable_stem_fusion=args.jit_disable_stem_fusion,
            disable_conv_bn_fusion=disable_conv_bn_fusion,
            experimental_conv_bn_fusion=(
                args.jit_experimental_conv_bn_fusion
                or variant == "jit_training_conv_bn_experimental"
            ),
        )

    return model


def time_one_batch(runner, optimizer, x, y):
    criterion = MSE()
    if hasattr(runner, "reset_train_profile"):
        runner.reset_train_profile()

    sync_device()
    batch_t0 = time.perf_counter()

    optimizer.zero_grad()

    sync_device()
    forward_t0 = time.perf_counter()
    pred = runner(x)
    loss = criterion(pred, y)
    sync_device()
    forward_sec = time.perf_counter() - forward_t0

    backward_t0 = time.perf_counter()
    loss.backward()
    sync_device()
    backward_sec = time.perf_counter() - backward_t0

    step_t0 = time.perf_counter()
    optimizer.step()
    sync_device()
    step_sec = time.perf_counter() - step_t0

    total_sec = time.perf_counter() - batch_t0
    if getattr(runner, "profile", False) and hasattr(runner, "print_train_backward_profile"):
        runner.print_train_backward_profile()

    return {
        "train_forward_ms": forward_sec * 1000.0,
        "backward_ms": backward_sec * 1000.0,
        "optimizer_step_ms": step_sec * 1000.0,
        "total_batch_ms": total_sec * 1000.0,
    }


def trimmed_mean(values, trim_ratio):
    values = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(values.size * trim_ratio)
    if trim > 0 and values.size > trim * 2:
        values = values[trim:-trim]
    return float(np.mean(values))


def summarize_timings(records, trim_ratio):
    keys = [
        "train_forward_ms",
        "backward_ms",
        "optimizer_step_ms",
        "total_batch_ms",
    ]
    summary = {}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_trimmed_mean"] = trimmed_mean(values, trim_ratio)
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_p50"] = float(np.percentile(values, 50))
        summary[f"{key}_p90"] = float(np.percentile(values, 90))
        summary[f"{key}_p95"] = float(np.percentile(values, 95))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))
    return summary


def run_variant(args, variant, device):
    clear_cupy_pool()
    runner = build_model(args, variant, device)
    optimizer = SGD(runner.parameters(), lr=args.lr)
    x, y = make_batch(args, device)

    for _ in range(args.warmup_batches):
        time_one_batch(runner, optimizer, x, y)

    records = []
    max_mem = memory_snapshot()
    for _ in range(args.measure_batches):
        record = time_one_batch(runner, optimizer, x, y)
        records.append(record)
        max_mem = max_memory(max_mem, memory_snapshot())

    summary = summarize_timings(records, args.trim_ratio)
    summary.update(max_mem)
    summary["variant"] = variant
    summary["device"] = device
    summary["warmup_batches"] = args.warmup_batches
    summary["measure_batches"] = args.measure_batches
    return summary, records


def resolve_variants(args):
    if args.variant_order:
        variants = [name.strip() for name in args.variant_order.split(",") if name.strip()]
    elif args.variant:
        variants = list(DEFAULT_VARIANTS) if "all" in args.variant else list(args.variant)
    else:
        variants = list(DEFAULT_VARIANTS)

    invalid = [name for name in variants if name not in VALID_VARIANTS]
    if invalid:
        raise ValueError(
            "unknown variant(s): "
            + ", ".join(invalid)
            + ". Valid variants: "
            + ", ".join(VALID_VARIANTS)
        )
    if len(set(variants)) != len(variants):
        raise ValueError("variant list/order contains duplicates")
    if not variants:
        raise ValueError("at least one variant must be selected")
    return variants


def default_output_name(args, variants):
    if args.output_name:
        return args.output_name
    if args.variant or args.variant_order:
        return "batch_breakdown_" + "__".join(variants) + ".json"
    return "batch_breakdown.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-level training breakdown for eager, dynamic fused, and JIT training backend."
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=VALID_VARIANTS + ("all",),
        default=None,
        help=(
            "Run one variant. Can be passed multiple times. "
            "Default is all variants in the default order."
        ),
    )
    parser.add_argument(
        "--variant-order",
        default=None,
        help=(
            "Comma-separated variant order, e.g. "
            "jit_training_backend,eager_original,eager_dynamic_fused. "
            "Overrides --variant."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measure-batches", type=int, default=5)
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=0.1,
        help="Fraction trimmed from each tail for trimmed_mean. Default: 0.1.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-seed", type=int, default=2026)
    parser.add_argument("--data-seed", type=int, default=2027)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dump-jit-graph", action="store_true")
    parser.add_argument(
        "--jit-profile",
        action="store_true",
        help="Enable per-node JIT training executor profiling for jit_training_backend.",
    )
    parser.add_argument(
        "--jit-disable-stem-fusion",
        action="store_true",
        help=(
            "For jit_training_backend, skip the first input-channel-3 Conv+BN+ReLU "
            "training fusion so the stem runs as Conv2dOp + fused BN/ReLU."
        ),
    )
    parser.add_argument(
        "--jit-disable-conv-bn-fusion",
        action="store_true",
        help=(
            "For JIT training variants, keep Conv2dOp separate and fuse only "
            "BN+ReLU / BN+Add+ReLU. This is the default training JIT strategy; "
            "the flag is retained for explicitness."
        ),
    )
    parser.add_argument(
        "--jit-experimental-conv-bn-fusion",
        action="store_true",
        help=(
            "Enable the experimental training Conv+BN+ReLU / Conv+BN+Add+ReLU "
            "large-fusion strategy. The jit_training_conv_bn_experimental "
            "variant enables this automatically."
        ),
    )
    parser.add_argument(
        "--cupy-accelerators",
        default=None,
        help=(
            "Value for CUPY_ACCELERATORS before importing CuPy. Defaults to an "
            "empty value to avoid CUB/NVCC reduction compilation; use 'auto' "
            "to leave CuPy defaults unchanged."
        ),
    )
    parser.add_argument(
        "--allow-unsupported-nvcc",
        action="store_true",
        help=(
            "Add -allow-unsupported-compiler to NVCC_PREPEND_FLAGS before "
            "importing CuPy. Use only if you intentionally want NVCC kernels "
            "with the installed MSVC toolchain."
        ),
    )
    parser.add_argument("--results-dir", default="results/jit_training_batch_breakdown")
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Output JSON file name inside --results-dir. Defaults to "
            "batch_breakdown.json for the default all-variant run, or a "
            "variant/order-specific file name when --variant/--variant-order is used."
        ),
    )

    args = parser.parse_args()
    if args.measure_batches < 1:
        parser.error("--measure-batches must be >= 1")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be >= 0")
    if not 0.0 <= args.trim_ratio < 0.5:
        parser.error("--trim-ratio must satisfy 0 <= trim_ratio < 0.5")
    try:
        args.resolved_variants = resolve_variants(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    device = "cuda" if cp is not None and not args.cpu else "cpu"
    args.effective_cupy_accelerators = os.environ.get("CUPY_ACCELERATORS")
    args.effective_nvcc_prepend_flags = os.environ.get("NVCC_PREPEND_FLAGS")
    variants = args.resolved_variants

    os.makedirs(args.results_dir, exist_ok=True)
    summaries = []
    all_records = {}

    print(f"Device: {device}")
    if device == "cuda":
        accelerators = os.environ.get("CUPY_ACCELERATORS")
        print(f"CuPy accelerators: {accelerators if accelerators else '<disabled>'}")
        nvcc_flags = os.environ.get("NVCC_PREPEND_FLAGS")
        if nvcc_flags:
            print(f"NVCC_PREPEND_FLAGS: {nvcc_flags}")
    print(
        "Timing phases: train_forward_ms includes model forward + MSE loss; "
        "total_batch_ms includes zero_grad, forward/loss, backward, and optimizer.step."
    )
    print("Variants: " + " -> ".join(variants))
    print(
        "Timing stats: mean, trimmed_mean, std, median/p50, p90, p95, min, max."
    )

    for variant in variants:
        print(f"\n=== {variant} ===")
        summary, records = run_variant(args, variant, device)
        summaries.append(summary)
        all_records[variant] = records
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    result = {
        "config": vars(args),
        "device": device,
        "variants": variants,
        "summaries": summaries,
        "records": all_records,
    }

    output_path = os.path.join(args.results_dir, default_output_name(args, variants))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
