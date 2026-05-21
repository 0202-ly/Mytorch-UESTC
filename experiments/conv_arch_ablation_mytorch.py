import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from math import gcd
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _argv_has(flag: str) -> bool:
    return flag in sys.argv


def _argv_value(flag: str) -> Optional[str]:
    prefix = flag + "="
    for idx, arg in enumerate(sys.argv):
        if arg == flag:
            return sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _preconfigure_cupy_compiler() -> None:
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
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cupy as cp
except ImportError:
    cp = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mytorch.dataloader import Dataloader  # noqa: E402
from mytorch.loss import MSELoss  # noqa: E402
from mytorch.modules import (  # noqa: E402
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Flatten,
    Linear,
    MaxPool,
    Module,
    ReLU,
)
from mytorch.optim import Adam  # noqa: E402
from mytorch.tensor import Tensor  # noqa: E402


DEFAULT_VARIANTS = [
    "standard_k3",
    "kernel1",
    "kernel5",
    "kernel31",
    "depthwise_k3",
    "group2_k3",
    "dilation2_k3",
    "transpose_k3",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if cp is not None:
        try:
            cp.random.seed(seed)
        except Exception:
            pass


def sync_device(device: str) -> None:
    if device == "cuda" and cp is not None:
        cp.cuda.Stream.null.synchronize()


def clear_cupy_pool() -> None:
    if cp is None:
        return
    sync_device("cuda")
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def gpu_used_mb() -> Optional[float]:
    if cp is None:
        return None
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return float((total_bytes - free_bytes) / (1024 ** 2))
    except Exception:
        return None


def cupy_pool_used_mb() -> Optional[float]:
    if cp is None:
        return None
    try:
        return float(cp.get_default_memory_pool().used_bytes() / (1024 ** 2))
    except Exception:
        return None


def stat_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
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


def to_float(value: Any) -> float:
    if hasattr(value, "data") and not isinstance(value, memoryview):
        value = value.data
    if cp is not None and isinstance(value, cp.ndarray):
        value = value.get()
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    return float(np.asarray(value).reshape(-1)[0])


def metric_from_arrays(pred: Any, target: Any) -> Tuple[float, float, int, int, int]:
    if cp is not None and isinstance(pred, cp.ndarray):
        pred = pred.get()
    if cp is not None and isinstance(target, cp.ndarray):
        target = target.get()
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    diff = pred_arr - target_arr
    abs_diff = np.abs(diff)
    return (
        float(np.sum(diff * diff)),
        float(np.sum(abs_diff)),
        int(diff.size),
        int(np.sum(abs_diff <= 0.05)),
        int(np.sum(abs_diff <= 0.10)),
    )


def summarize_metric_sums(sse: float, sae: float, n: int, acc05: int, acc10: int) -> Dict[str, float]:
    denom = max(1, int(n))
    mse = float(sse / denom)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(sae / denom),
        "acc_at_0.05": float(acc05 / denom),
        "acc_at_0.10": float(acc10 / denom),
    }


def read_list(path: str, limit: Optional[int] = None) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, label = line.split()[:2]
            items.append((rel, float(label)))
            if limit is not None and len(items) >= limit:
                break
    return items


def resolve_list_path(data_root: str, list_path: str) -> str:
    if os.path.isabs(list_path):
        return list_path
    candidate = os.path.join(data_root, list_path)
    if os.path.exists(candidate):
        return candidate
    return list_path


class MyTorchDonkeyDataset:
    def __init__(
        self,
        data_root: str,
        list_path: str,
        image_height: int,
        image_width: int,
        limit: Optional[int] = None,
    ):
        if cv2 is None and Image is None:
            raise ImportError("opencv-python or Pillow is required for image loading.")
        self.data_root = data_root
        self.list_path = resolve_list_path(data_root, list_path)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.items = read_list(self.list_path, limit=limit)
        if not self.items:
            raise RuntimeError(f"No samples found in {self.list_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        rel, steering = self.items[int(idx)]
        path = os.path.join(self.data_root, rel)
        if cv2 is not None:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (self.image_height, self.image_width):
                img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        else:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("RGB")
                if pil_img.size != (self.image_width, self.image_height):
                    pil_img = pil_img.resize((self.image_width, self.image_height), Image.BILINEAR)
                img = np.asarray(pil_img)
        img_chw = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
        target = np.asarray([steering], dtype=np.float32)
        return Tensor(img_chw), Tensor(target)


def parse_variant(name: str) -> Dict[str, Any]:
    configs = {
        "standard_k3": {"kernel_size": 3, "conv_kind": "standard", "dilation": 1, "groups": 1},
        "kernel1": {"kernel_size": 1, "conv_kind": "standard", "dilation": 1, "groups": 1},
        "kernel5": {"kernel_size": 5, "conv_kind": "standard", "dilation": 1, "groups": 1},
        "kernel31": {"kernel_size": 31, "conv_kind": "standard", "dilation": 1, "groups": 1},
        "depthwise_k3": {"kernel_size": 3, "conv_kind": "depthwise", "dilation": 1, "groups": 1},
        "group2_k3": {"kernel_size": 3, "conv_kind": "grouped", "dilation": 1, "groups": 2},
        "dilation2_k3": {"kernel_size": 3, "conv_kind": "standard", "dilation": 2, "groups": 1},
        "transpose_k3": {"kernel_size": 3, "conv_kind": "transpose", "dilation": 1, "groups": 1},
    }
    if name not in configs:
        raise ValueError(f"Unknown variant {name}. Valid variants: {', '.join(configs)}")
    config = dict(configs[name])
    config["name"] = name
    return config


def same_padding(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1) // 2


class MyTorchDepthwiseSeparableConv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.depthwise = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.depthwise_bn = BatchNorm2d(in_channels)
        self.depthwise_relu = ReLU()
        self.pointwise = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)
        return self.pointwise(x)


def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    conv_kind: str,
    dilation: int,
    groups: int,
) -> Module:
    padding = same_padding(kernel_size, dilation)
    if conv_kind == "depthwise":
        return MyTorchDepthwiseSeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
    if conv_kind == "grouped":
        actual_groups = groups if in_channels % groups == 0 and out_channels % groups == 0 else gcd(in_channels, out_channels)
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=max(1, actual_groups),
        )
    if conv_kind == "transpose" and stride == 1:
        return ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
    return Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
    )


class AblationBasicBlock(Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        conv_kind: str,
        dilation: int,
        groups: int,
    ):
        super().__init__()
        self.conv1 = make_conv(in_channels, out_channels, kernel_size, stride, conv_kind, dilation, groups)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu1 = ReLU()
        self.conv2 = make_conv(out_channels, out_channels, kernel_size, 1, conv_kind, dilation, groups)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu2 = ReLU()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.downsample_bn = BatchNorm2d(out_channels)
            self.downsample = True

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample_bn(self.downsample_conv(x))
        out = out + identity
        return self.relu2(out)


class DonkeyResNet18AblationMyTorch(Module):
    def __init__(
        self,
        kernel_size: int,
        conv_kind: str,
        dilation: int,
        groups: int,
        base_channels: int = 32,
        output_dim: int = 1,
    ):
        super().__init__()
        self.in_channels = base_channels
        self.base_channels = int(base_channels)
        self.stem_conv = Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3)
        self.stem_bn = BatchNorm2d(base_channels)
        self.stem_relu = ReLU()
        self.maxpool = MaxPool(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base_channels, 2, 1, kernel_size, conv_kind, dilation, groups)
        self.layer2 = self._make_layer(base_channels * 2, 2, 2, kernel_size, conv_kind, dilation, groups)
        self.layer3 = self._make_layer(base_channels * 4, 2, 2, kernel_size, conv_kind, dilation, groups)
        self.layer4 = self._make_layer(base_channels * 8, 2, 2, kernel_size, conv_kind, dilation, groups)
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = Linear(base_channels * 8, output_dim)

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
        kernel_size: int,
        conv_kind: str,
        dilation: int,
        groups: int,
    ) -> List[Module]:
        layers: List[Module] = [
            AblationBasicBlock(self.in_channels, out_channels, stride, kernel_size, conv_kind, dilation, groups)
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(AblationBasicBlock(self.in_channels, out_channels, 1, kernel_size, conv_kind, dilation, groups))
        return layers

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.maxpool(x)
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


def count_parameters(model: Module) -> int:
    total = 0
    for param in model.parameters():
        total += int(np.prod(param.shape()))
    return total


def conv2d_out(hw: Tuple[int, int], kernel: int, stride: int, padding: int, dilation: int = 1) -> Tuple[int, int]:
    h, w = hw
    oh = (h + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    ow = (w + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    return int(oh), int(ow)


def pool_out(hw: Tuple[int, int], kernel: int, stride: int, padding: int) -> Tuple[int, int]:
    return conv2d_out(hw, kernel=kernel, stride=stride, padding=padding)


def estimate_flops(
    image_height: int,
    image_width: int,
    base_channels: int,
    kernel_size: int,
    conv_kind: str,
    dilation: int,
    groups: int,
    output_dim: int,
) -> int:
    hw = (int(image_height), int(image_width))
    flops = 0

    def add_conv(in_c: int, out_c: int, kernel: int, stride: int = 1, padding: int = 0, dilation_value: int = 1, group_count: int = 1) -> Tuple[int, int]:
        nonlocal hw, flops
        oh, ow = conv2d_out(hw, kernel, stride, padding, dilation_value)
        flops += oh * ow * out_c * (in_c // group_count) * kernel * kernel * 2
        hw = (oh, ow)
        return hw

    def add_conv_at(hw_in: Tuple[int, int], in_c: int, out_c: int, kernel: int, stride: int, padding: int, dilation_value: int, group_count: int) -> Tuple[int, Tuple[int, int]]:
        oh, ow = conv2d_out(hw_in, kernel, stride, padding, dilation_value)
        ops = oh * ow * out_c * (in_c // group_count) * kernel * kernel * 2
        return int(ops), (oh, ow)

    add_conv(3, base_channels, 7, stride=2, padding=3)
    hw = pool_out(hw, 3, stride=2, padding=1)

    current_channels = base_channels
    for out_channels, first_stride in [
        (base_channels, 1),
        (base_channels * 2, 2),
        (base_channels * 4, 2),
        (base_channels * 8, 2),
    ]:
        for block_idx in range(2):
            stride = first_stride if block_idx == 0 else 1
            block_input_hw = hw
            block_in_channels = current_channels
            padding = same_padding(kernel_size, dilation)

            if conv_kind == "depthwise":
                ops, new_hw = add_conv_at(block_input_hw, block_in_channels, block_in_channels, kernel_size, stride, padding, dilation, block_in_channels)
                flops += ops
                ops, _ = add_conv_at(new_hw, block_in_channels, out_channels, 1, 1, 0, 1, 1)
                flops += ops
                hw = new_hw
            elif conv_kind == "grouped":
                actual_groups = groups if block_in_channels % groups == 0 and out_channels % groups == 0 else gcd(block_in_channels, out_channels)
                ops, hw = add_conv_at(block_input_hw, block_in_channels, out_channels, kernel_size, stride, padding, dilation, max(1, actual_groups))
                flops += ops
            else:
                ops, hw = add_conv_at(block_input_hw, block_in_channels, out_channels, kernel_size, stride, padding, dilation, 1)
                flops += ops

            if conv_kind == "depthwise":
                ops, new_hw = add_conv_at(hw, out_channels, out_channels, kernel_size, 1, padding, dilation, out_channels)
                flops += ops
                ops, _ = add_conv_at(new_hw, out_channels, out_channels, 1, 1, 0, 1, 1)
                flops += ops
                hw = new_hw
            elif conv_kind == "grouped":
                actual_groups = groups if out_channels % groups == 0 else 1
                ops, hw = add_conv_at(hw, out_channels, out_channels, kernel_size, 1, padding, dilation, max(1, actual_groups))
                flops += ops
            else:
                ops, hw = add_conv_at(hw, out_channels, out_channels, kernel_size, 1, padding, dilation, 1)
                flops += ops

            if stride != 1 or block_in_channels != out_channels:
                ops, _ = add_conv_at(block_input_hw, block_in_channels, out_channels, 1, stride, 0, 1, 1)
                flops += ops
            current_channels = out_channels

    flops += base_channels * 8 * output_dim * 2
    return int(flops)


def evaluate(model: Module, loader: Dataloader, criterion: MSELoss, args: argparse.Namespace) -> Dict[str, Any]:
    model.eval()
    sse = 0.0
    sae = 0.0
    elems = 0
    acc05 = 0
    acc10 = 0
    batches = 0
    samples = 0
    sync_device(args.device)
    t0 = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        if args.max_val_batches and batch_idx > args.max_val_batches:
            break
        x, y = batch
        if args.device == "cuda":
            x = x.cuda()
            y = y.cuda()
        pred = model(x)
        _ = criterion(pred, y)
        batch_sse, batch_sae, batch_elems, batch_acc05, batch_acc10 = metric_from_arrays(pred.data, y.data)
        sse += batch_sse
        sae += batch_sae
        elems += batch_elems
        acc05 += batch_acc05
        acc10 += batch_acc10
        samples += int(x.shape()[0])
        batches += 1
    sync_device(args.device)
    elapsed = time.perf_counter() - t0
    out = summarize_metric_sums(sse, sae, elems, acc05, acc10)
    out.update({
        "samples": samples,
        "batches": batches,
        "time_sec": elapsed,
        "samples_per_sec": samples / elapsed if elapsed > 0 else None,
    })
    return out


def measure_forward_latency(model: Module, loader: Dataloader, args: argparse.Namespace) -> Dict[str, Optional[float]]:
    model.eval()
    iterator = iter(loader)
    try:
        x, _ = next(iterator)
    except StopIteration:
        return {"latency_ms_p50": None, "latency_ms_mean": None, "latency_ms_p90": None}
    if args.device == "cuda":
        x = x.cuda()
    for _ in range(args.latency_warmup):
        model(x)
    sync_device(args.device)
    values: List[float] = []
    for _ in range(args.latency_repeats):
        t0 = time.perf_counter()
        model(x)
        sync_device(args.device)
        values.append((time.perf_counter() - t0) * 1000.0)
    stats = stat_summary(values)
    return {
        "latency_ms_p50": stats["p50"],
        "latency_ms_mean": stats["mean"],
        "latency_ms_p90": stats["p90"],
    }


def train_variant(
    variant: str,
    args: argparse.Namespace,
    train_ds: MyTorchDonkeyDataset,
    val_ds: MyTorchDonkeyDataset,
    output_dir: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    variant_config = parse_variant(variant)
    clear_cupy_pool()
    set_seed(args.seed)
    train_loader = Dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = Dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    model = DonkeyResNet18AblationMyTorch(
        kernel_size=variant_config["kernel_size"],
        conv_kind=variant_config["conv_kind"],
        dilation=variant_config["dilation"],
        groups=variant_config["groups"],
        base_channels=args.base_channels,
        output_dim=args.output_dim,
    )
    if args.device == "cuda":
        model.cuda()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MSELoss(fused=False)
    history: List[Dict[str, Any]] = []
    best_val = None
    best_epoch = None
    batch_times: List[float] = []
    peak_gpu = gpu_used_mb()
    peak_pool = cupy_pool_used_mb()
    total_train_time = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sse = 0.0
        train_sae = 0.0
        train_elems = 0
        train_acc05 = 0
        train_acc10 = 0
        train_samples = 0
        epoch_t0 = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader, start=1):
            if args.max_train_batches and batch_idx > args.max_train_batches:
                break
            x, y = batch
            if args.device == "cuda":
                x = x.cuda()
                y = y.cuda()
            sync_device(args.device)
            batch_t0 = time.perf_counter()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            sync_device(args.device)
            batch_times.append((time.perf_counter() - batch_t0) * 1000.0)

            batch_sse, batch_sae, batch_elems, batch_acc05, batch_acc10 = metric_from_arrays(pred.data, y.data)
            train_sse += batch_sse
            train_sae += batch_sae
            train_elems += batch_elems
            train_acc05 += batch_acc05
            train_acc10 += batch_acc10
            train_samples += int(x.shape()[0])
            gpu_now = gpu_used_mb()
            pool_now = cupy_pool_used_mb()
            peak_gpu = max([v for v in (peak_gpu, gpu_now) if v is not None], default=None)
            peak_pool = max([v for v in (peak_pool, pool_now) if v is not None], default=None)

        train_time = time.perf_counter() - epoch_t0
        total_train_time += train_time
        val_metrics = evaluate(model, val_loader, criterion, args)
        train_metrics = summarize_metric_sums(train_sse, train_sae, train_elems, train_acc05, train_acc10)
        train_metrics.update({
            "samples": train_samples,
            "time_sec": train_time,
            "samples_per_sec": train_samples / train_time if train_time > 0 else None,
        })
        record = {
            "variant": variant,
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_acc_at_0.10": train_metrics["acc_at_0.10"],
            "train_time_sec": train_time,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_acc_at_0.10": val_metrics["acc_at_0.10"],
            "val_time_sec": val_metrics["time_sec"],
        }
        history.append(record)
        if best_val is None or val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_epoch = epoch
        print(
            f"[{variant}] epoch={epoch:02d} "
            f"train_mse={train_metrics['mse']:.6f} val_mse={val_metrics['mse']:.6f}"
        )

    latency_loader = Dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    latency = measure_forward_latency(model, latency_loader, args)
    batch_stats = stat_summary(batch_times)
    flops = estimate_flops(
        args.image_height,
        args.image_width,
        args.base_channels,
        variant_config["kernel_size"],
        variant_config["conv_kind"],
        variant_config["dilation"],
        variant_config["groups"],
        args.output_dim,
    )
    summary = {
        "variant": variant,
        "framework": "MyTorch",
        "kernel_size": variant_config["kernel_size"],
        "conv_kind": variant_config["conv_kind"],
        "dilation": variant_config["dilation"],
        "groups": variant_config["groups"],
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "final_val_mse": history[-1]["val_mse"] if history else None,
        "final_val_mae": history[-1]["val_mae"] if history else None,
        "final_val_acc_at_0.10": history[-1]["val_acc_at_0.10"] if history else None,
        "avg_train_time_sec": total_train_time / max(1, args.epochs),
        "batch_ms_p50": batch_stats["p50"],
        "batch_ms_p90": batch_stats["p90"],
        "latency_ms_p50": latency["latency_ms_p50"],
        "latency_ms_p90": latency["latency_ms_p90"],
        "params": count_parameters(model),
        "flops": flops,
        "gflops": flops / 1e9,
        "peak_gpu_used_mb": peak_gpu,
        "peak_cupy_pool_used_mb": peak_pool,
    }
    return summary, history


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_md(path: str, summaries: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    sorted_rows = sorted(summaries, key=lambda item: item.get("best_val_mse") if item.get("best_val_mse") is not None else float("inf"))

    def fmt(value: Any, digits: int = 5) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.{digits}g}"
        return str(value)

    lines = [
        "# MyTorch Conv Architecture Ablation",
        "",
        "Framework: MyTorch. This table must replace the previous PyTorch conv architecture ablation when reporting framework results.",
        "",
        f"- Train list: `{args.train_list}`",
        f"- Val list: `{args.val_list}`",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Base channels: {args.base_channels}",
        f"- Device: {args.device}",
        "",
        "| variant | best MSE | acc@0.10 | GFLOPs | latency p50 ms | batch p50 ms | peak GPU MB | params |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["variant"]),
                fmt(row.get("best_val_mse")),
                fmt(row.get("final_val_acc_at_0.10")),
                fmt(row.get("gflops")),
                fmt(row.get("latency_ms_p50")),
                fmt(row.get("batch_ms_p50")),
                fmt(row.get("peak_gpu_used_mb")),
                str(row.get("params")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "Notes:",
        "- All variants are implemented with MyTorch modules and trained through MyTorch autograd/optimizer.",
        "- `transpose_k3` uses ConvTranspose2d only for stride=1 residual convolutions; stride=2 downsampling keeps Conv2d to preserve feature map size.",
        "- FLOPs are estimated analytically for one 120x160 image and use multiply-add as two FLOPs.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_val_curves(path: str, history: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in history:
        by_variant.setdefault(row["variant"], []).append(row)
    plt.figure(figsize=(10, 6))
    for variant, rows in by_variant.items():
        rows = sorted(rows, key=lambda item: item["epoch"])
        plt.plot([row["epoch"] for row in rows], [row["val_mse"] for row in rows], marker="o", linewidth=1.5, label=variant)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("MyTorch Conv Architecture Ablation")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MyTorch DonkeyCar convolution architecture ablation.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--val-list", default="val.txt")
    parser.add_argument("--results-dir", default="results/conv_arch_ablation_mytorch")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--allow-unsupported-nvcc", action="store_true")
    parser.add_argument(
        "--cupy-accelerators",
        default=None,
        help="none/off/disabled disables CuPy accelerators; auto leaves CuPy defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if cp is not None else "cpu"
    if args.device == "cuda" and cp is None:
        raise RuntimeError("MyTorch CUDA run requires CuPy in the active Python environment.")
    args.max_train_batches = args.max_train_batches or None
    args.max_val_batches = args.max_val_batches or None
    args.train_limit = args.train_limit or None
    args.val_limit = args.val_limit or None

    run_dir = os.path.join(args.results_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_ds = MyTorchDonkeyDataset(args.data_root, args.train_list, args.image_height, args.image_width, args.train_limit)
    val_ds = MyTorchDonkeyDataset(args.data_root, args.val_list, args.image_height, args.image_width, args.val_limit)

    summaries: List[Dict[str, Any]] = []
    all_history: List[Dict[str, Any]] = []
    for variant in args.variants:
        summary, history = train_variant(variant, args, train_ds, val_ds, run_dir)
        summaries.append(summary)
        all_history.extend(history)

    write_csv(os.path.join(run_dir, "summary.csv"), summaries)
    write_csv(os.path.join(run_dir, "history.csv"), all_history)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    write_summary_md(os.path.join(run_dir, "summary.md"), summaries, args)
    plot_val_curves(os.path.join(run_dir, "val_loss_curves.png"), all_history)

    print(f"Saved MyTorch conv architecture ablation to: {run_dir}")
    print(os.path.join(run_dir, "summary.md"))


if __name__ == "__main__":
    main()
