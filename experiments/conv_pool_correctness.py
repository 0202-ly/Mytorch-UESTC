import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("PyTorch is required for this experiment.") from exc

from mytorch.function import AvgPoolOp, Conv2dOp, MaxPoolOp
from mytorch.tensor import Tensor


Array = np.ndarray
Pair = Tuple[int, int]


def as_pair(value: Any) -> Pair:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def out_dim(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def max_abs(a: Array, b: Array) -> float:
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def conv2d_naive_forward(
    x: Array,
    w: Array,
    b: Optional[Array],
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Array:
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    dh, dw = as_pair(dilation)
    n, c, h, width = x.shape
    out_c, c_per_group, kh, kw = w.shape
    if c % groups != 0 or out_c % groups != 0:
        raise ValueError("channels must be divisible by groups")
    if c_per_group != c // groups:
        raise ValueError("weight input channels do not match groups")

    oh = out_dim(h, kh, sh, ph, dh)
    ow = out_dim(width, kw, sw, pw, dw)
    if oh <= 0 or ow <= 0:
        raise ValueError("invalid output shape")

    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    out = np.zeros((n, out_c, oh, ow), dtype=np.float64)
    out_per_group = out_c // groups

    for ni in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for ocg in range(out_per_group):
                oc = o0 + ocg
                for oy in range(oh):
                    iy0 = oy * sh
                    for ox in range(ow):
                        ix0 = ox * sw
                        acc = 0.0
                        for icg in range(c_per_group):
                            ic = c0 + icg
                            for ky in range(kh):
                                iy = iy0 + ky * dh
                                for kx in range(kw):
                                    ix = ix0 + kx * dw
                                    acc += x_pad[ni, ic, iy, ix] * w[oc, icg, ky, kx]
                        if b is not None:
                            acc += b[oc]
                        out[ni, oc, oy, ox] = acc
    return out


def conv2d_naive_backward(
    x: Array,
    w: Array,
    b: Optional[Array],
    dout: Array,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tuple[Array, Array, Optional[Array]]:
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    dh, dw = as_pair(dilation)
    n, c, h, width = x.shape
    out_c, c_per_group, kh, kw = w.shape
    _, _, oh, ow = dout.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    dx_pad = np.zeros_like(x_pad, dtype=np.float64)
    dw_arr = np.zeros_like(w, dtype=np.float64)
    db_arr = np.sum(dout, axis=(0, 2, 3)) if b is not None else None
    out_per_group = out_c // groups

    for ni in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for ocg in range(out_per_group):
                oc = o0 + ocg
                for oy in range(oh):
                    iy0 = oy * sh
                    for ox in range(ow):
                        ix0 = ox * sw
                        grad = dout[ni, oc, oy, ox]
                        for icg in range(c_per_group):
                            ic = c0 + icg
                            for ky in range(kh):
                                iy = iy0 + ky * dh
                                for kx in range(kw):
                                    ix = ix0 + kx * dw
                                    dw_arr[oc, icg, ky, kx] += x_pad[ni, ic, iy, ix] * grad
                                    dx_pad[ni, ic, iy, ix] += w[oc, icg, ky, kx] * grad

    dx = dx_pad[:, :, ph:ph + h, pw:pw + width]
    return dx, dw_arr, db_arr


def pool2d_naive_forward(
    x: Array,
    kernel_size: Any,
    stride: Any,
    padding: Any = 0,
    mode: str = "max",
) -> Array:
    kh, kw = as_pair(kernel_size)
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    n, c, h, width = x.shape
    oh = out_dim(h, kh, sh, ph, 1)
    ow = out_dim(width, kw, sw, pw, 1)
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    out = np.zeros((n, c, oh, ow), dtype=np.float64)

    for ni in range(n):
        for ci in range(c):
            for oy in range(oh):
                iy0 = oy * sh
                for ox in range(ow):
                    ix0 = ox * sw
                    window = x_pad[ni, ci, iy0:iy0 + kh, ix0:ix0 + kw]
                    if mode == "max":
                        out[ni, ci, oy, ox] = np.max(window)
                    elif mode == "avg":
                        out[ni, ci, oy, ox] = np.mean(window)
                    else:
                        raise ValueError(f"unknown pool mode: {mode}")
    return out


def pool2d_naive_backward(
    x: Array,
    dout: Array,
    kernel_size: Any,
    stride: Any,
    padding: Any = 0,
    mode: str = "max",
) -> Array:
    kh, kw = as_pair(kernel_size)
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    n, c, h, width = x.shape
    _, _, oh, ow = dout.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    dx_pad = np.zeros_like(x_pad, dtype=np.float64)

    for ni in range(n):
        for ci in range(c):
            for oy in range(oh):
                iy0 = oy * sh
                for ox in range(ow):
                    ix0 = ox * sw
                    if mode == "avg":
                        dx_pad[ni, ci, iy0:iy0 + kh, ix0:ix0 + kw] += dout[ni, ci, oy, ox] / (kh * kw)
                    else:
                        window = x_pad[ni, ci, iy0:iy0 + kh, ix0:ix0 + kw]
                        flat_idx = int(np.argmax(window))
                        ky, kx = divmod(flat_idx, kw)
                        dx_pad[ni, ci, iy0 + ky, ix0 + kx] += dout[ni, ci, oy, ox]

    return dx_pad[:, :, ph:ph + h, pw:pw + width]


def numerical_grad_inplace(arr: Array, fn: Callable[[], float], eps: float) -> Array:
    grad = np.zeros_like(arr, dtype=np.float64)
    it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(arr[idx])
        arr[idx] = old + eps
        f_pos = fn()
        arr[idx] = old - eps
        f_neg = fn()
        arr[idx] = old
        grad[idx] = (f_pos - f_neg) / (2.0 * eps)
        it.iternext()
    return grad


def conv_numeric_grads(
    x: Array,
    w: Array,
    b: Array,
    dout: Array,
    params: Dict[str, Any],
    eps: float,
) -> Tuple[Array, Array, Array]:
    x_num = x.copy()
    w_num = w.copy()
    b_num = b.copy()

    def scalar(xv: Array, wv: Array, bv: Array) -> float:
        out = conv2d_naive_forward(xv, wv, bv, **params)
        return float(np.sum(out * dout))

    dx = numerical_grad_inplace(x_num, lambda: scalar(x_num, w, b), eps)
    dw = numerical_grad_inplace(w_num, lambda: scalar(x, w_num, b), eps)
    db = numerical_grad_inplace(b_num, lambda: scalar(x, w, b_num), eps)
    return dx, dw, db


def pool_numeric_grad(
    x: Array,
    dout: Array,
    params: Dict[str, Any],
    eps: float,
) -> Array:
    x_num = x.copy()

    def scalar() -> float:
        out = pool2d_naive_forward(x_num, **params)
        return float(np.sum(out * dout))

    return numerical_grad_inplace(x_num, scalar, eps)


def torch_conv_forward_backward(
    x: Array,
    w: Array,
    b: Array,
    dout: Array,
    params: Dict[str, Any],
) -> Tuple[Array, Array, Array, Array]:
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    out = F.conv2d(
        xt,
        wt,
        bt,
        stride=params["stride"],
        padding=params["padding"],
        dilation=params["dilation"],
        groups=params["groups"],
    )
    out.backward(torch.tensor(dout, dtype=torch.float64))
    return (
        out.detach().cpu().numpy(),
        xt.grad.detach().cpu().numpy(),
        wt.grad.detach().cpu().numpy(),
        bt.grad.detach().cpu().numpy(),
    )


def mytorch_conv_forward_backward(
    x: Array,
    w: Array,
    b: Array,
    dout: Array,
    params: Dict[str, Any],
) -> Tuple[Array, Array, Array, Array]:
    xt = Tensor(x.copy(), requires_grad=True)
    wt = Tensor(w.copy(), requires_grad=True)
    bt = Tensor(b.reshape(1, -1).copy(), requires_grad=True)
    op = Conv2dOp()
    out = op(
        xt,
        wt,
        bt,
        stride=params["stride"],
        padding=params["padding"],
        dilation=params["dilation"],
        groups=params["groups"],
    )
    op.backward(dout.copy())
    return out.data.copy(), xt.grad.copy(), wt.grad.copy(), bt.grad.reshape(-1).copy()


def torch_pool_forward_backward(
    x: Array,
    dout: Array,
    params: Dict[str, Any],
) -> Tuple[Array, Array]:
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    mode = params["mode"]
    ph, pw = as_pair(params["padding"])
    pool_input = xt
    if ph or pw:
        # Match MyTorch's explicit zero-padding semantics. PyTorch's built-in
        # max_pool2d padding uses negative infinity, so manual padding keeps the
        # three implementations comparable.
        pool_input = F.pad(xt, (pw, pw, ph, ph), mode="constant", value=0.0)
    if mode == "max":
        out = F.max_pool2d(
            pool_input,
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=0,
        )
    elif mode == "avg":
        out = F.avg_pool2d(
            pool_input,
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=0,
            count_include_pad=True,
        )
    else:
        raise ValueError(f"unknown pool mode: {mode}")
    out.backward(torch.tensor(dout, dtype=torch.float64))
    return out.detach().cpu().numpy(), xt.grad.detach().cpu().numpy()


def mytorch_pool_forward_backward(
    x: Array,
    dout: Array,
    params: Dict[str, Any],
) -> Tuple[Array, Array]:
    xt = Tensor(x.copy(), requires_grad=True)
    op = MaxPoolOp() if params["mode"] == "max" else AvgPoolOp()
    out = op(xt, params["kernel_size"], params["stride"], params["padding"])
    op.backward(dout.copy())
    return out.data.copy(), xt.grad.copy()


def random_non_tie_positive(rng: np.random.Generator, shape: Sequence[int]) -> Array:
    base = rng.uniform(0.1, 2.0, size=shape)
    offsets = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) * 1e-4
    return base + offsets


def build_conv_cases() -> List[Dict[str, Any]]:
    return [
        {
            "name": "conv_standard_pad1",
            "shape": (2, 2, 5, 6),
            "out_channels": 3,
            "kernel_size": (3, 3),
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
        },
        {
            "name": "conv_asym_stride_pad",
            "shape": (2, 3, 7, 6),
            "out_channels": 4,
            "kernel_size": (3, 2),
            "stride": (2, 1),
            "padding": (1, 0),
            "dilation": 1,
            "groups": 1,
        },
        {
            "name": "conv_dilation2",
            "shape": (1, 2, 7, 8),
            "out_channels": 3,
            "kernel_size": (3, 3),
            "stride": 1,
            "padding": 2,
            "dilation": 2,
            "groups": 1,
        },
        {
            "name": "conv_groups2",
            "shape": (2, 4, 6, 5),
            "out_channels": 4,
            "kernel_size": (3, 3),
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 2,
        },
        {
            "name": "conv_depthwise_stride2",
            "shape": (1, 4, 7, 7),
            "out_channels": 4,
            "kernel_size": (3, 3),
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "groups": 4,
        },
        {
            "name": "conv_grouped_dilated_asym",
            "shape": (1, 4, 8, 9),
            "out_channels": 6,
            "kernel_size": (2, 3),
            "stride": (2, 2),
            "padding": (1, 2),
            "dilation": (2, 1),
            "groups": 2,
        },
    ]


def build_pool_cases() -> List[Dict[str, Any]]:
    return [
        {
            "name": "maxpool_k2_s2",
            "mode": "max",
            "shape": (2, 3, 6, 6),
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
        },
        {
            "name": "maxpool_k3_s1_pad1",
            "mode": "max",
            "shape": (1, 2, 5, 7),
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
        {
            "name": "avgpool_k2_s2",
            "mode": "avg",
            "shape": (2, 2, 6, 5),
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
        },
        {
            "name": "avgpool_k3_s1_pad1",
            "mode": "avg",
            "shape": (1, 3, 5, 6),
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
        {
            "name": "avgpool_asym_stride",
            "mode": "avg",
            "shape": (1, 2, 7, 8),
            "kernel_size": (2, 3),
            "stride": (2, 1),
            "padding": (0, 1),
        },
    ]


def run_conv_case(case: Dict[str, Any], rng: np.random.Generator, eps: float, threshold: float) -> Dict[str, Any]:
    n, in_channels, h, w_in = case["shape"]
    groups = int(case["groups"])
    out_channels = int(case["out_channels"])
    kh, kw = as_pair(case["kernel_size"])
    c_per_group = in_channels // groups
    x = rng.normal(0.0, 0.7, size=case["shape"]).astype(np.float64)
    weight = rng.normal(0.0, 0.4, size=(out_channels, c_per_group, kh, kw)).astype(np.float64)
    bias = rng.normal(0.0, 0.2, size=(out_channels,)).astype(np.float64)
    params = {
        "stride": case["stride"],
        "padding": case["padding"],
        "dilation": case["dilation"],
        "groups": groups,
    }

    out_np = conv2d_naive_forward(x, weight, bias, **params)
    dout = rng.normal(0.0, 0.5, size=out_np.shape).astype(np.float64)
    dx_np, dw_np, db_np = conv2d_naive_backward(x, weight, bias, dout, **params)

    out_torch, dx_torch, dw_torch, db_torch = torch_conv_forward_backward(x, weight, bias, dout, params)
    out_mt, dx_mt, dw_mt, db_mt = mytorch_conv_forward_backward(x, weight, bias, dout, params)

    dx_num, dw_num, db_num = conv_numeric_grads(x, weight, bias, dout, params, eps)

    mt_backward_max = max(max_abs(dx_mt, dx_np), max_abs(dw_mt, dw_np), max_abs(db_mt, db_np))
    torch_backward_max = max(max_abs(dx_torch, dx_np), max_abs(dw_torch, dw_np), max_abs(db_torch, db_np))
    mt_gradcheck_max = max(max_abs(dx_mt, dx_num), max_abs(dw_mt, dw_num), max_abs(db_mt, db_num))
    np_gradcheck_max = max(max_abs(dx_np, dx_num), max_abs(dw_np, dw_num), max_abs(db_np, db_num))

    worst = max(
        max_abs(out_mt, out_np),
        max_abs(out_torch, out_np),
        mt_backward_max,
        torch_backward_max,
        mt_gradcheck_max,
        np_gradcheck_max,
    )
    return {
        "op": "conv2d",
        "case": case["name"],
        "input_shape": str(tuple(case["shape"])),
        "output_shape": str(tuple(out_np.shape)),
        "kernel_size": str(as_pair(case["kernel_size"])),
        "stride": str(as_pair(case["stride"])),
        "padding": str(as_pair(case["padding"])),
        "dilation": str(as_pair(case["dilation"])),
        "groups": groups,
        "mytorch_forward_max_abs": max_abs(out_mt, out_np),
        "pytorch_forward_max_abs": max_abs(out_torch, out_np),
        "mytorch_backward_max_abs": mt_backward_max,
        "pytorch_backward_max_abs": torch_backward_max,
        "mytorch_dx_max_abs": max_abs(dx_mt, dx_np),
        "mytorch_dw_max_abs": max_abs(dw_mt, dw_np),
        "mytorch_db_max_abs": max_abs(db_mt, db_np),
        "pytorch_dx_max_abs": max_abs(dx_torch, dx_np),
        "pytorch_dw_max_abs": max_abs(dw_torch, dw_np),
        "pytorch_db_max_abs": max_abs(db_torch, db_np),
        "mytorch_gradcheck_max_abs": mt_gradcheck_max,
        "numpy_gradcheck_max_abs": np_gradcheck_max,
        "gradcheck_dx_max_abs": max_abs(dx_mt, dx_num),
        "gradcheck_dw_max_abs": max_abs(dw_mt, dw_num),
        "gradcheck_db_max_abs": max_abs(db_mt, db_num),
        "threshold": threshold,
        "status": "PASS" if worst <= threshold else "FAIL",
    }


def run_pool_case(case: Dict[str, Any], rng: np.random.Generator, eps: float, threshold: float) -> Dict[str, Any]:
    if case["mode"] == "max":
        x = random_non_tie_positive(rng, case["shape"]).astype(np.float64)
    else:
        x = rng.normal(0.0, 0.7, size=case["shape"]).astype(np.float64)
    params = {
        "kernel_size": case["kernel_size"],
        "stride": case["stride"],
        "padding": case["padding"],
        "mode": case["mode"],
    }
    out_np = pool2d_naive_forward(x, **params)
    dout = rng.normal(0.0, 0.5, size=out_np.shape).astype(np.float64)
    dx_np = pool2d_naive_backward(x, dout, **params)

    out_torch, dx_torch = torch_pool_forward_backward(x, dout, params)
    out_mt, dx_mt = mytorch_pool_forward_backward(x, dout, params)
    dx_num = pool_numeric_grad(x, dout, params, eps)

    mt_backward_max = max_abs(dx_mt, dx_np)
    torch_backward_max = max_abs(dx_torch, dx_np)
    mt_gradcheck_max = max_abs(dx_mt, dx_num)
    np_gradcheck_max = max_abs(dx_np, dx_num)
    worst = max(
        max_abs(out_mt, out_np),
        max_abs(out_torch, out_np),
        mt_backward_max,
        torch_backward_max,
        mt_gradcheck_max,
        np_gradcheck_max,
    )
    return {
        "op": f"{case['mode']}pool2d",
        "case": case["name"],
        "input_shape": str(tuple(case["shape"])),
        "output_shape": str(tuple(out_np.shape)),
        "kernel_size": str(as_pair(case["kernel_size"])),
        "stride": str(as_pair(case["stride"])),
        "padding": str(as_pair(case["padding"])),
        "dilation": "-",
        "groups": "-",
        "mytorch_forward_max_abs": max_abs(out_mt, out_np),
        "pytorch_forward_max_abs": max_abs(out_torch, out_np),
        "mytorch_backward_max_abs": mt_backward_max,
        "pytorch_backward_max_abs": torch_backward_max,
        "mytorch_dx_max_abs": mt_backward_max,
        "mytorch_dw_max_abs": "",
        "mytorch_db_max_abs": "",
        "pytorch_dx_max_abs": torch_backward_max,
        "pytorch_dw_max_abs": "",
        "pytorch_db_max_abs": "",
        "mytorch_gradcheck_max_abs": mt_gradcheck_max,
        "numpy_gradcheck_max_abs": np_gradcheck_max,
        "gradcheck_dx_max_abs": mt_gradcheck_max,
        "gradcheck_dw_max_abs": "",
        "gradcheck_db_max_abs": "",
        "threshold": threshold,
        "status": "PASS" if worst <= threshold else "FAIL",
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "op",
        "case",
        "input_shape",
        "output_shape",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "mytorch_forward_max_abs",
        "pytorch_forward_max_abs",
        "mytorch_backward_max_abs",
        "pytorch_backward_max_abs",
        "mytorch_dx_max_abs",
        "mytorch_dw_max_abs",
        "mytorch_db_max_abs",
        "pytorch_dx_max_abs",
        "pytorch_dw_max_abs",
        "pytorch_db_max_abs",
        "mytorch_gradcheck_max_abs",
        "numpy_gradcheck_max_abs",
        "gradcheck_dx_max_abs",
        "gradcheck_dw_max_abs",
        "gradcheck_db_max_abs",
        "threshold",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return f"{value:.3e}"
    return str(value)


def markdown_table(rows: List[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> List[str]:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for _, key in columns) + " |")
    return lines


def write_markdown(path: str, rows: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    passed = sum(1 for row in rows if row["status"] == "PASS")
    lines = [
        "# Conv/Pool Correctness Experiment",
        "",
        "Baseline: NumPy naive implementation. PyTorch and MyTorch are compared against the same forward output and upstream-gradient backward result.",
        "Pooling comparisons use MyTorch-compatible explicit zero padding in the PyTorch path.",
        "",
        f"- Cases: {len(rows)}",
        f"- Passed: {passed}/{len(rows)}",
        f"- Epsilon for central difference: {config['eps']}",
        f"- Pass threshold: {config['threshold']}",
        f"- PyTorch version: {torch.__version__}",
        "",
        "## Summary",
        "",
    ]
    summary_cols = [
        ("op", "op"),
        ("case", "case"),
        ("stride", "stride"),
        ("padding", "padding"),
        ("dilation", "dilation"),
        ("groups", "groups"),
        ("MyTorch fwd", "mytorch_forward_max_abs"),
        ("MyTorch bwd", "mytorch_backward_max_abs"),
        ("MyTorch gradcheck", "mytorch_gradcheck_max_abs"),
        ("PyTorch fwd", "pytorch_forward_max_abs"),
        ("PyTorch bwd", "pytorch_backward_max_abs"),
        ("status", "status"),
    ]
    lines.extend(markdown_table(rows, summary_cols))
    lines.extend(
        [
            "",
            "## Detailed Gradients",
            "",
        ]
    )
    detail_cols = [
        ("op", "op"),
        ("case", "case"),
        ("mt dx", "mytorch_dx_max_abs"),
        ("mt dw", "mytorch_dw_max_abs"),
        ("mt db", "mytorch_db_max_abs"),
        ("torch dx", "pytorch_dx_max_abs"),
        ("torch dw", "pytorch_dw_max_abs"),
        ("torch db", "pytorch_db_max_abs"),
        ("gc dx", "gradcheck_dx_max_abs"),
        ("gc dw", "gradcheck_dw_max_abs"),
        ("gc db", "gradcheck_db_max_abs"),
    ]
    lines.extend(markdown_table(rows, detail_cols))
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Conv2d/Pool2d correctness experiment.")
    parser.add_argument("--results-dir", default=os.path.join("results", "conv_pool_correctness"))
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=1e-7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for case in build_conv_cases():
        rows.append(run_conv_case(case, rng, args.eps, args.threshold))
    for case in build_pool_cases():
        rows.append(run_pool_case(case, rng, args.eps, args.threshold))

    config = {
        "seed": args.seed,
        "eps": args.eps,
        "threshold": args.threshold,
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_csv(os.path.join(output_dir, "results.csv"), rows)
    write_markdown(os.path.join(output_dir, "summary.md"), rows, config)

    print(f"Saved correctness results to: {output_dir}")
    print(f"Passed: {sum(1 for row in rows if row['status'] == 'PASS')}/{len(rows)}")
    print(os.path.join(output_dir, "summary.md"))
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
