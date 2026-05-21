import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import psutil
except ImportError:
    psutil = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("PyTorch is required for this experiment.") from exc

from mytorch.function import Conv2dOp
from mytorch.tensor import Tensor, GPU_AVAILABLE as MYTORCH_GPU_AVAILABLE


Pair = Tuple[int, int]


def as_pair(value: Any) -> Pair:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def out_dim(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def conv_output_shape(
    input_shape: Sequence[int],
    out_channels: int,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
) -> Tuple[int, int, int, int]:
    n, _, h, w = input_shape
    kh, kw = as_pair(kernel_size)
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    dh, dw = as_pair(dilation)
    oh = out_dim(h, kh, sh, ph, dh)
    ow = out_dim(w, kw, sw, pw, dw)
    return n, out_channels, oh, ow


def conv_forward_flops(
    input_shape: Sequence[int],
    out_channels: int,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
) -> int:
    n, in_channels, _, _ = input_shape
    _, _, oh, ow = conv_output_shape(input_shape, out_channels, kernel_size, stride, padding, dilation)
    kh, kw = as_pair(kernel_size)
    macs = n * out_channels * oh * ow * (in_channels // groups) * kh * kw
    return int(2 * macs)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.data
    if cp is not None and isinstance(value, cp.ndarray):
        return value.get()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def naive_conv2d_forward(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
) -> np.ndarray:
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    dh, dw = as_pair(dilation)
    n, in_channels, h, width = x.shape
    out_channels, c_per_group, kh, kw = w.shape
    _, _, oh, ow = conv_output_shape(x.shape, out_channels, (kh, kw), stride, padding, dilation)
    out_per_group = out_channels // groups
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    out = np.empty((n, out_channels, oh, ow), dtype=np.float32)

    for ni in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for ocg in range(out_per_group):
                oc = o0 + ocg
                kernel = w[oc]
                bias = 0.0 if b is None else float(b[oc])
                for oy in range(oh):
                    iy0 = oy * sh
                    for ox in range(ow):
                        ix0 = ox * sw
                        acc = 0.0
                        for ky in range(kh):
                            iy = iy0 + ky * dh
                            for kx in range(kw):
                                ix = ix0 + kx * dw
                                acc += float(np.dot(x_pad[ni, c0:c0 + c_per_group, iy, ix], kernel[:, ky, kx]))
                        out[ni, oc, oy, ox] = acc + bias
    return out


def sync_device(device: str) -> None:
    if device == "cuda":
        if cp is not None:
            cp.cuda.Stream.null.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def process_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def clear_gpu_memory() -> None:
    if cp is not None:
        try:
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cupy_pool_total_mb() -> Optional[float]:
    if cp is None:
        return None
    try:
        return cp.get_default_memory_pool().total_bytes() / (1024 ** 2)
    except Exception:
        return None


def cupy_pool_used_mb() -> Optional[float]:
    if cp is None:
        return None
    try:
        return cp.get_default_memory_pool().used_bytes() / (1024 ** 2)
    except Exception:
        return None


def torch_peak_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        return None


def time_call(
    fn: Callable[[], Any],
    warmup: int,
    repeats: int,
    device: str,
) -> Tuple[float, List[float], Any]:
    result = None
    for _ in range(warmup):
        result = fn()
    sync_device(device)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        sync_device(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(np.asarray(times, dtype=np.float64))), times, result


def run_method(
    method: str,
    device: str,
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    params: Dict[str, Any],
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    clear_gpu_memory()
    rss_before = process_rss_mb()
    gpu_before_cupy_total = cupy_pool_total_mb()

    if method == "numpy_naive":
        def fn():
            return naive_conv2d_forward(x, w, b, **params)

    elif method == "mytorch_im2col_gemm":
        xt = Tensor(x.copy(), requires_grad=False)
        wt = Tensor(w.copy(), requires_grad=False)
        bt = Tensor(b.reshape(1, -1).copy(), requires_grad=False)
        if device == "cuda":
            xt.cuda()
            wt.cuda()
            bt.cuda()

        def fn():
            return Conv2dOp()(xt, wt, bt, **params)

    elif method == "pytorch_conv2d":
        torch_device = torch.device("cuda" if device == "cuda" else "cpu")
        xt = torch.from_numpy(x).to(torch_device)
        wt = torch.from_numpy(w).to(torch_device)
        bt = torch.from_numpy(b).to(torch_device)

        def fn():
            return F.conv2d(
                xt,
                wt,
                bt,
                stride=params["stride"],
                padding=params["padding"],
                dilation=params["dilation"],
                groups=params["groups"],
            )

    else:
        raise ValueError(f"unknown method: {method}")

    latency_ms, samples, result = time_call(fn, warmup=warmup, repeats=repeats, device=device)
    rss_after = process_rss_mb()
    gpu_after_cupy_total = cupy_pool_total_mb()
    return {
        "method": method,
        "device": device,
        "latency_ms_p50": latency_ms,
        "latency_ms_mean": float(np.mean(samples)),
        "latency_ms_p90": float(np.percentile(samples, 90)),
        "repeats": repeats,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": None if rss_before is None or rss_after is None else rss_after - rss_before,
        "cupy_pool_used_mb": cupy_pool_used_mb() if method.startswith("mytorch") and device == "cuda" else None,
        "cupy_pool_total_mb": gpu_after_cupy_total if method.startswith("mytorch") and device == "cuda" else None,
        "cupy_pool_total_delta_mb": (
            None
            if gpu_before_cupy_total is None or gpu_after_cupy_total is None or not method.startswith("mytorch") or device != "cuda"
            else gpu_after_cupy_total - gpu_before_cupy_total
        ),
        "torch_peak_allocated_mb": torch_peak_mb() if method.startswith("pytorch") and device == "cuda" else None,
        "output": to_numpy(result),
    }


def build_cases() -> List[Dict[str, Any]]:
    return [
        {
            "name": "batch1_rgb_fmap16_k3",
            "input_shape": (1, 3, 16, 16),
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "naive_repeats": 3,
        },
        {
            "name": "batch8_rgb_fmap16_k3",
            "input_shape": (8, 3, 16, 16),
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "naive_repeats": 1,
        },
        {
            "name": "channels16_fmap16_k3",
            "input_shape": (4, 16, 16, 16),
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "naive_repeats": 1,
        },
        {
            "name": "channels32_fmap16_k1",
            "input_shape": (4, 32, 16, 16),
            "out_channels": 64,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "naive_repeats": 1,
        },
        {
            "name": "kernel5_fmap24",
            "input_shape": (2, 8, 24, 24),
            "out_channels": 16,
            "kernel_size": 5,
            "stride": 1,
            "padding": 2,
            "dilation": 1,
            "groups": 1,
            "naive_repeats": 1,
        },
        {
            "name": "fmap32_channels8_k3",
            "input_shape": (2, 8, 32, 32),
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "naive_repeats": 1,
        },
        {
            "name": "stride2_fmap48_k3",
            "input_shape": (2, 8, 48, 48),
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "naive_repeats": 1,
        },
        {
            "name": "groups4_channels16_k3",
            "input_shape": (2, 16, 24, 24),
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 4,
            "naive_repeats": 1,
        },
    ]


def method_specs(include_cuda: bool) -> List[Dict[str, Any]]:
    specs = [
        {"method": "numpy_naive", "device": "cpu", "warmup": 0, "repeats": None},
        {"method": "mytorch_im2col_gemm", "device": "cpu", "warmup": 3, "repeats": 20},
        {"method": "pytorch_conv2d", "device": "cpu", "warmup": 5, "repeats": 30},
    ]
    if include_cuda:
        specs.extend(
            [
                {"method": "mytorch_im2col_gemm", "device": "cuda", "warmup": 8, "repeats": 50},
                {"method": "pytorch_conv2d", "device": "cuda", "warmup": 10, "repeats": 80},
            ]
        )
    return specs


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        if value == 0:
            return "0"
        return f"{value:.3e}"
    return str(value)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "case",
        "method",
        "device",
        "input_shape",
        "output_shape",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "forward_flops",
        "latency_ms_p50",
        "latency_ms_mean",
        "latency_ms_p90",
        "gflops_per_sec",
        "rss_delta_mb",
        "cupy_pool_used_mb",
        "cupy_pool_total_mb",
        "torch_peak_allocated_mb",
        "max_abs_vs_numpy_naive",
        "repeats",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: List[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> List[str]:
    lines = [
        "| " + " | ".join(name for name, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    return lines


def write_markdown(path: str, rows: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    cols = [
        ("case", "case"),
        ("method", "method"),
        ("device", "device"),
        ("lat ms p50", "latency_ms_p50"),
        ("GFLOP/s", "gflops_per_sec"),
        ("RSS delta MB", "rss_delta_mb"),
        ("CuPy pool MB", "cupy_pool_total_mb"),
        ("Torch peak MB", "torch_peak_allocated_mb"),
        ("max abs err", "max_abs_vs_numpy_naive"),
    ]
    lines = [
        "# Conv2d Performance Experiment",
        "",
        "Methods: NumPy naive forward, MyTorch Conv2dOp im2col+GEMM, and PyTorch F.conv2d.",
        "FLOPs are analytic forward FLOPs: 2 * N * Cout * OH * OW * (Cin/groups) * KH * KW.",
        "",
        f"- Python: {config['python']}",
        f"- NumPy: {config['numpy_version']}",
        f"- PyTorch: {config['torch_version']}",
        f"- CuPy: {config['cupy_version']}",
        f"- MyTorch GPU_AVAILABLE: {config['mytorch_gpu_available']}",
        f"- CUDA device: {config['cuda_device']}",
        "",
        "## Results",
        "",
    ]
    lines.extend(markdown_table(rows, cols))
    lines.extend(["", "## Speedup Summary", ""])

    speed_rows = []
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[f"{row['method']}:{row['device']}"] = row
    for case_name, group in by_case.items():
        naive = group.get("numpy_naive:cpu")
        mt_cpu = group.get("mytorch_im2col_gemm:cpu")
        mt_cuda = group.get("mytorch_im2col_gemm:cuda")
        torch_cpu = group.get("pytorch_conv2d:cpu")
        torch_cuda = group.get("pytorch_conv2d:cuda")
        speed_rows.append(
            {
                "case": case_name,
                "naive_div_mytorch_cpu": (
                    naive["latency_ms_p50"] / mt_cpu["latency_ms_p50"] if naive and mt_cpu else None
                ),
                "naive_div_pytorch_cpu": (
                    naive["latency_ms_p50"] / torch_cpu["latency_ms_p50"] if naive and torch_cpu else None
                ),
                "mytorch_cpu_div_cuda": (
                    mt_cpu["latency_ms_p50"] / mt_cuda["latency_ms_p50"] if mt_cpu and mt_cuda else None
                ),
                "mytorch_cuda_div_pytorch_cuda": (
                    mt_cuda["latency_ms_p50"] / torch_cuda["latency_ms_p50"] if mt_cuda and torch_cuda else None
                ),
            }
        )
    speed_cols = [
        ("case", "case"),
        ("naive / mt CPU", "naive_div_mytorch_cpu"),
        ("naive / torch CPU", "naive_div_pytorch_cpu"),
        ("mt CPU / mt CUDA", "mytorch_cpu_div_cuda"),
        ("mt CUDA / torch CUDA", "mytorch_cuda_div_pytorch_cuda"),
    ]
    lines.extend(markdown_table(speed_rows, speed_cols))
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_case(case: Dict[str, Any], specs: List[Dict[str, Any]], rng: np.random.Generator) -> List[Dict[str, Any]]:
    in_shape = tuple(case["input_shape"])
    groups = int(case["groups"])
    out_channels = int(case["out_channels"])
    kh, kw = as_pair(case["kernel_size"])
    c_per_group = in_shape[1] // groups
    x = rng.normal(0.0, 1.0, size=in_shape).astype(np.float32)
    w = rng.normal(0.0, 0.2, size=(out_channels, c_per_group, kh, kw)).astype(np.float32)
    b = rng.normal(0.0, 0.1, size=(out_channels,)).astype(np.float32)
    params = {
        "stride": case["stride"],
        "padding": case["padding"],
        "dilation": case["dilation"],
        "groups": groups,
    }
    output_shape = conv_output_shape(in_shape, out_channels, case["kernel_size"], case["stride"], case["padding"], case["dilation"])
    flops = conv_forward_flops(in_shape, out_channels, case["kernel_size"], case["stride"], case["padding"], case["dilation"], groups)

    rows = []
    baseline = None
    for spec in specs:
        repeats = case["naive_repeats"] if spec["method"] == "numpy_naive" else spec["repeats"]
        result = run_method(
            method=spec["method"],
            device=spec["device"],
            x=x,
            w=w,
            b=b,
            params=params,
            warmup=spec["warmup"],
            repeats=int(repeats),
        )
        output = result.pop("output")
        if spec["method"] == "numpy_naive":
            baseline = output
        if baseline is None:
            baseline = naive_conv2d_forward(x, w, b, **params)

        latency_sec = result["latency_ms_p50"] / 1000.0
        row = {
            "case": case["name"],
            "input_shape": str(in_shape),
            "output_shape": str(output_shape),
            "out_channels": out_channels,
            "kernel_size": str(as_pair(case["kernel_size"])),
            "stride": str(as_pair(case["stride"])),
            "padding": str(as_pair(case["padding"])),
            "dilation": str(as_pair(case["dilation"])),
            "groups": groups,
            "forward_flops": flops,
            "gflops_per_sec": flops / latency_sec / 1e9 if latency_sec > 0 else None,
            "max_abs_vs_numpy_naive": max_abs(output, baseline),
            **result,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Conv2d performance benchmark.")
    parser.add_argument("--results-dir", default=os.path.join("results", "conv_performance"))
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    include_cuda = (
        not args.cpu_only
        and MYTORCH_GPU_AVAILABLE
        and cp is not None
        and torch.cuda.is_available()
    )
    rng = np.random.default_rng(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    specs = method_specs(include_cuda=include_cuda)
    rows: List[Dict[str, Any]] = []
    for case in build_cases():
        print(f"Running {case['name']}...")
        rows.extend(run_case(case, specs, rng))

    config = {
        "seed": args.seed,
        "python": sys.executable,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cupy_version": None if cp is None else cp.__version__,
        "mytorch_gpu_available": MYTORCH_GPU_AVAILABLE,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "include_cuda": include_cuda,
        "kmp_duplicate_lib_ok": os.environ.get("KMP_DUPLICATE_LIB_OK"),
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_csv(os.path.join(output_dir, "results.csv"), rows)
    write_markdown(os.path.join(output_dir, "summary.md"), rows, config)

    print(f"Saved performance results to: {output_dir}")
    print(os.path.join(output_dir, "summary.md"))


if __name__ == "__main__":
    main()
