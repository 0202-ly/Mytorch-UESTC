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


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.augmentation_ablation import (  # noqa: E402
    DonkeyRegressionDataset,
    plot_validation_loss_curves,
    resolve_list_path,
    write_csv,
)


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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_variant(name: str) -> Dict[str, Any]:
    configs = {
        "standard_k3": {
            "kernel_size": 3,
            "conv_kind": "standard",
            "dilation": 1,
            "groups": 1,
            "description": "baseline standard Conv2d with 3x3 residual-block kernels",
        },
        "kernel1": {
            "kernel_size": 1,
            "conv_kind": "standard",
            "dilation": 1,
            "groups": 1,
            "description": "replace residual-block kernels with 1x1 Conv2d",
        },
        "kernel5": {
            "kernel_size": 5,
            "conv_kind": "standard",
            "dilation": 1,
            "groups": 1,
            "description": "replace residual-block kernels with 5x5 Conv2d",
        },
        "kernel31": {
            "kernel_size": 31,
            "conv_kind": "standard",
            "dilation": 1,
            "groups": 1,
            "description": "large-kernel 31x31 Conv2d in residual blocks",
        },
        "depthwise_k3": {
            "kernel_size": 3,
            "conv_kind": "depthwise",
            "dilation": 1,
            "groups": 1,
            "description": "depthwise separable convolution: depthwise 3x3 + pointwise 1x1",
        },
        "group2_k3": {
            "kernel_size": 3,
            "conv_kind": "grouped",
            "dilation": 1,
            "groups": 2,
            "description": "grouped 3x3 Conv2d with groups=2 where channel counts allow it",
        },
        "dilation2_k3": {
            "kernel_size": 3,
            "conv_kind": "standard",
            "dilation": 2,
            "groups": 1,
            "description": "dilated 3x3 Conv2d with dilation=2 in residual blocks",
        },
        "transpose_k3": {
            "kernel_size": 3,
            "conv_kind": "transpose",
            "dilation": 1,
            "groups": 1,
            "description": "ConvTranspose2d used for stride-1 residual-block convolutions; downsample convolutions stay standard",
        },
    }
    if name not in configs:
        raise ValueError(f"Unknown variant: {name}. Available: {', '.join(configs)}")
    return configs[name]


def same_padding(kernel_size: int, dilation: int = 1) -> int:
    return ((kernel_size - 1) * dilation) // 2


def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    conv_kind: str,
    dilation: int,
    groups: int,
) -> nn.Module:
    padding = same_padding(kernel_size, dilation)
    if conv_kind == "depthwise":
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
    if conv_kind == "grouped":
        actual_groups = groups if in_channels % groups == 0 and out_channels % groups == 0 else gcd(in_channels, out_channels)
        actual_groups = max(1, actual_groups)
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=actual_groups,
            bias=False,
        )
    if conv_kind == "transpose" and stride == 1:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


class AblationBasicBlock(nn.Module):
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
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = make_conv(out_channels, out_channels, kernel_size, 1, conv_kind, dilation, groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class DonkeyResNet18Ablation(nn.Module):
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
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_channels, 2, 1, kernel_size, conv_kind, dilation, groups)
        self.layer2 = self._make_layer(base_channels * 2, 2, 2, kernel_size, conv_kind, dilation, groups)
        self.layer3 = self._make_layer(base_channels * 4, 2, 2, kernel_size, conv_kind, dilation, groups)
        self.layer4 = self._make_layer(base_channels * 8, 2, 2, kernel_size, conv_kind, dilation, groups)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, output_dim)

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
        kernel_size: int,
        conv_kind: str,
        dilation: int,
        groups: int,
    ) -> nn.Sequential:
        layers = [
            AblationBasicBlock(self.in_channels, out_channels, stride, kernel_size, conv_kind, dilation, groups)
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(AblationBasicBlock(self.in_channels, out_channels, 1, kernel_size, conv_kind, dilation, groups))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sse = 0.0
    sae = 0.0
    acc05 = 0
    acc10 = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        diff = pred - y
        abs_diff = torch.abs(diff)
        sse += float(torch.sum(diff * diff).detach().cpu().item())
        sae += float(torch.sum(abs_diff).detach().cpu().item())
        acc05 += int(torch.sum(abs_diff <= 0.05).detach().cpu().item())
        acc10 += int(torch.sum(abs_diff <= 0.10).detach().cpu().item())
        n += int(y.numel())
    mse = sse / max(1, n)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": sae / max(1, n),
        "acc_at_0.05": acc05 / max(1, n),
        "acc_at_0.10": acc10 / max(1, n),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_flops(model: nn.Module, input_shape: Tuple[int, int, int, int], device: torch.device) -> int:
    hooks = []
    flops = {"total": 0}

    def hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            out = output
            batch = int(out.shape[0])
            out_c = int(out.shape[1])
            out_h = int(out.shape[2])
            out_w = int(out.shape[3])
            k_h, k_w = module.kernel_size
            groups = getattr(module, "groups", 1)
            in_c = int(module.in_channels)
            flops["total"] += batch * out_c * out_h * out_w * (in_c // groups) * k_h * k_w * 2
        elif isinstance(module, nn.Linear):
            out = output
            batch = int(out.shape[0])
            flops["total"] += batch * int(module.in_features) * int(module.out_features) * 2

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(input_shape, device=device)
        model(dummy)
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return int(flops["total"])


@torch.no_grad()
def measure_forward_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Dict[str, Optional[float]]:
    model.eval()
    try:
        x, _ = next(iter(loader))
    except StopIteration:
        return {"latency_ms_p50": None, "latency_ms_mean": None, "latency_ms_p90": None}
    x = x.to(device, non_blocking=True)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_mean": float(np.mean(arr)),
        "latency_ms_p90": float(np.percentile(arr, 90)),
    }


def train_variant(
    variant: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cfg = parse_variant(variant)
    model = DonkeyResNet18Ablation(
        kernel_size=cfg["kernel_size"],
        conv_kind=cfg["conv_kind"],
        dilation=cfg["dilation"],
        groups=cfg["groups"],
        base_channels=args.base_channels,
    ).to(device)

    flops_per_sample = estimate_model_flops(model, (1, 3, args.image_height, args.image_width), device)
    params = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best = {
        "epoch": None,
        "mse": float("inf"),
        "rmse": None,
        "mae": None,
        "acc_at_0.05": None,
        "acc_at_0.10": None,
    }
    history: List[Dict[str, Any]] = []
    total_t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_t0 = time.perf_counter()
        sse = 0.0
        sae = 0.0
        n = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            if args.max_train_batches is not None and batch_idx > args.max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                diff = pred.detach() - y
                sse += float(torch.sum(diff * diff).detach().cpu().item())
                sae += float(torch.sum(torch.abs(diff)).detach().cpu().item())
                n += int(y.numel())

        val_metrics = evaluate(model, val_loader, device)
        train_mse = sse / max(1, n)
        row = {
            "variant": variant,
            "epoch": epoch,
            "train_mse": train_mse,
            "train_rmse": float(np.sqrt(train_mse)),
            "train_mae": sae / max(1, n),
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_acc_at_0.05": val_metrics["acc_at_0.05"],
            "val_acc_at_0.10": val_metrics["acc_at_0.10"],
            "epoch_time_sec": time.perf_counter() - epoch_t0,
        }
        history.append(row)
        if val_metrics["mse"] < best["mse"]:
            best = {"epoch": epoch, **val_metrics}
        print(
            f"[{variant}] epoch={epoch:02d} "
            f"train_mse={row['train_mse']:.6f} val_mse={row['val_mse']:.6f} "
            f"acc@0.10={row['val_acc_at_0.10']:.4f}"
        )

    latency = measure_forward_latency(model, val_loader, device, args.latency_warmup, args.latency_repeats)
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    summary = {
        "variant": variant,
        "description": cfg["description"],
        "conv_kind": cfg["conv_kind"],
        "kernel_size": cfg["kernel_size"],
        "dilation": cfg["dilation"],
        "groups": cfg["groups"],
        "base_channels": args.base_channels,
        "params": params,
        "flops_per_sample": flops_per_sample,
        "gflops_per_sample": flops_per_sample / 1e9,
        "best_epoch": best["epoch"],
        "best_val_mse": best["mse"],
        "best_val_rmse": best["rmse"],
        "best_val_mae": best["mae"],
        "best_val_acc_at_0.05": best["acc_at_0.05"],
        "best_val_acc_at_0.10": best["acc_at_0.10"],
        "final_val_mse": history[-1]["val_mse"],
        "total_time_sec": time.perf_counter() - total_t0,
        "peak_memory_mb": peak_memory_mb,
        **latency,
    }
    if args.save_models:
        torch.save(model.state_dict(), os.path.join(output_dir, f"{variant}_state.pt"))
    return summary, history


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: str, summaries: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> None:
    lines = [
        "# DonkeyCar Convolution Form Ablation",
        "",
        "Task: steering-angle regression with a ResNet18-style PyTorch model. Only the residual-block convolution form changes across variants.",
        "",
        f"- Device: {config['device']}",
        f"- Epochs: {config['epochs']}",
        f"- Batch size: {config['batch_size']}",
        f"- Base channels: {config['base_channels']}",
        f"- Train list: `{config['train_list']}`",
        f"- Val list: `{config['val_list']}`",
        "",
        "## Summary",
        "",
        "| variant | conv | k | dilation | groups | params | GFLOPs/sample | best val MSE | acc@0.05 | acc@0.10 | p50 latency ms | peak MB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda item: item["best_val_mse"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    row["conv_kind"],
                    fmt(row["kernel_size"]),
                    fmt(row["dilation"]),
                    fmt(row["groups"]),
                    fmt(row["params"]),
                    fmt(row["gflops_per_sample"]),
                    fmt(row["best_val_mse"]),
                    fmt(row["best_val_acc_at_0.05"]),
                    fmt(row["best_val_acc_at_0.10"]),
                    fmt(row["latency_ms_p50"]),
                    fmt(row["peak_memory_mb"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Variant Notes",
            "",
        ]
    )
    for row in summaries:
        lines.append(f"- `{row['variant']}`: {row['description']}.")
    lines.extend(
        [
            "",
            "Validation loss curves: `val_loss_curves.png`.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="DonkeyCar convolution-form ablation for ResNet18-style models.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--val-list", default="val.txt")
    parser.add_argument("--results-dir", default=os.path.join("results", "conv_arch_ablation"))
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    train_ds = DonkeyRegressionDataset(
        args.data_root,
        "train",
        transform=None,
        limit=args.max_train_samples,
        train=False,
        list_path=args.train_list,
    )
    val_ds = DonkeyRegressionDataset(
        args.data_root,
        "val",
        transform=None,
        limit=args.max_val_samples,
        train=False,
        list_path=args.val_list,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    summaries: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    for idx, variant in enumerate(args.variants):
        set_seed(args.seed + idx)
        summary, history = train_variant(variant, args, train_loader, val_loader, device, output_dir)
        summaries.append(summary)
        history_rows.extend(history)

    config = {
        **vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "output_dir": output_dir,
        "train_list": resolve_list_path(args.data_root, args.train_list),
        "val_list": resolve_list_path(args.data_root, args.val_list),
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"config": config, "summaries": summaries}, f, indent=2, ensure_ascii=False)
    write_csv(os.path.join(output_dir, "summary.csv"), summaries)
    write_csv(os.path.join(output_dir, "history.csv"), history_rows)
    plot_validation_loss_curves(history_rows, os.path.join(output_dir, "val_loss_curves.png"))
    write_markdown(os.path.join(output_dir, "summary.md"), summaries, config)

    print(f"Saved convolution ablation results to: {output_dir}")
    print(os.path.join(output_dir, "summary.md"))
    print(os.path.join(output_dir, "val_loss_curves.png"))


if __name__ == "__main__":
    main()
