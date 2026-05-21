import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.autodrive_net_pytorch import AutoDriveNetPyTorch
from mytorch.transforms import (
    build_donkeycar_batch_transform,
    build_donkeycar_transform,
    describe_donkeycar_augmentation,
)


IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


class DonkeyRegressionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        transform,
        limit: Optional[int],
        train: bool,
        list_path: Optional[str] = None,
    ):
        self.data_root = data_root
        self.transform = transform
        self.train = bool(train)
        default_name = "train.txt" if split == "train" else "val.txt"
        self.list_path = resolve_list_path(data_root, list_path or default_name)
        self.items = read_list(self.list_path, limit=limit)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rel, label = self.items[idx]
        img = load_rgb(os.path.join(self.data_root, rel))
        if self.train and self.transform is not None:
            img, label = self.transform(img, label)
        img_chw = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
        target = np.asarray([label], dtype=np.float32)
        return torch.from_numpy(img_chw), torch.from_numpy(target)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sse = 0.0
    sae = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        diff = pred - y
        sse += float(torch.sum(diff * diff).detach().cpu().item())
        sae += float(torch.sum(torch.abs(diff)).detach().cpu().item())
        n += int(y.numel())
    mse = sse / max(1, n)
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": sae / max(1, n)}


def train_variant(
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    train_transform = build_donkeycar_transform(
        variant,
        crop_steering_gain=args.crop_steering_gain,
        rotate_steering_gain=args.rotate_steering_gain,
        use_random_crop=args.use_random_crop,
        use_random_rotation=args.use_random_rotation,
        center_crop_height_ratio=args.center_crop_height_ratio,
        center_crop_width_ratio=args.center_crop_width_ratio,
    )
    batch_transform = build_donkeycar_batch_transform(
        variant,
        mix_alpha=args.mix_alpha,
        cutmix_alpha=args.cutmix_alpha,
        local_mix_alpha=args.local_mix_alpha,
        local_mix_max_diff=args.local_mix_max_diff,
    )
    train_ds = DonkeyRegressionDataset(
        args.data_root,
        "train",
        transform=train_transform,
        limit=args.max_train_samples,
        train=True,
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

    model = AutoDriveNetPyTorch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    history: List[Dict[str, Any]] = []
    best = {"epoch": None, "mse": float("inf"), "rmse": None, "mae": None}
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_sse = 0.0
        epoch_sae = 0.0
        epoch_n = 0
        epoch_t0 = time.perf_counter()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if batch_transform is not None:
                x, y = batch_transform(x, y)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                diff = pred.detach() - y
                epoch_sse += float(torch.sum(diff * diff).cpu().item())
                epoch_sae += float(torch.sum(torch.abs(diff)).cpu().item())
                epoch_n += int(y.numel())

        val_metrics = evaluate(model, val_loader, device)
        train_mse = epoch_sse / max(1, epoch_n)
        row = {
            "variant": variant,
            "epoch": epoch,
            "train_mse": train_mse,
            "train_rmse": float(np.sqrt(train_mse)),
            "train_mae": epoch_sae / max(1, epoch_n),
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "epoch_time_sec": time.perf_counter() - epoch_t0,
        }
        history.append(row)
        if val_metrics["mse"] < best["mse"]:
            best = {"epoch": epoch, **val_metrics}
        print(
            f"[{variant}] epoch={epoch:02d} "
            f"train_mse={row['train_mse']:.6f} val_mse={row['val_mse']:.6f}"
        )

    total_time = time.perf_counter() - t0
    final = history[-1]
    summary = {
        "variant": variant,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_list": train_ds.list_path,
        "val_list": val_ds.list_path,
        "epochs": args.epochs,
        "total_time_sec": total_time,
        "best_epoch": best["epoch"],
        "best_val_mse": best["mse"],
        "best_val_rmse": best["rmse"],
        "best_val_mae": best["mae"],
        "final_val_mse": final["val_mse"],
        "final_val_rmse": final["val_rmse"],
        "final_val_mae": final["val_mae"],
    }
    torch.save(model.state_dict(), os.path.join(output_dir, f"{variant}_state.pt"))
    return summary, history


def tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    arr = np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def make_sample_grid(
    data_root: str,
    train_list: str,
    output_path: str,
    seed: int,
    samples_per_variant: int = 5,
    crop_steering_gain: float = 0.0,
    rotate_steering_gain: float = 0.0,
    use_random_crop: bool = False,
    use_random_rotation: bool = False,
    center_crop_height_ratio: float = 0.85,
    center_crop_width_ratio: float = 1.0,
    mix_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    local_mix_alpha: float = 0.2,
    local_mix_max_diff: float = 0.05,
    variants: Optional[Sequence[str]] = None,
) -> None:
    np.random.seed(seed)
    random.seed(seed)
    items = read_list(resolve_list_path(data_root, train_list), limit=max(16, samples_per_variant * 2))
    variants = list(variants or [
        "none",
        "center_crop",
        "flip_only",
        "brightness_only",
        "contrast_only",
        "gamma_only",
        "noise_only",
        "hsv_only",
        "mixup_only",
        "local_mixup_only",
        "cutmix_only",
    ])
    cell_h = IMAGE_HEIGHT
    cell_w = IMAGE_WIDTH
    label_h = 24
    variant_w = 178
    grid = np.full(
        (len(variants) * (cell_h + label_h), variant_w + samples_per_variant * cell_w, 3),
        255,
        dtype=np.uint8,
    )

    for r, variant in enumerate(variants):
        image_transform = build_donkeycar_transform(
            "basic" if variant in {"mixup", "cutmix"} else variant,
            crop_steering_gain=crop_steering_gain,
            rotate_steering_gain=rotate_steering_gain,
            use_random_crop=use_random_crop,
            use_random_rotation=use_random_rotation,
            center_crop_height_ratio=center_crop_height_ratio,
            center_crop_width_ratio=center_crop_width_ratio,
        )
        batch_transform = build_donkeycar_batch_transform(
            variant,
            mix_alpha=mix_alpha,
            cutmix_alpha=cutmix_alpha,
            local_mix_alpha=local_mix_alpha,
            local_mix_max_diff=local_mix_max_diff,
        )
        imgs = []
        labels = []
        for i in range(samples_per_variant):
            rel, label = items[i]
            img = load_rgb(os.path.join(data_root, rel))
            img, label = image_transform(img, label)
            imgs.append(torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)))
            labels.append(torch.tensor([label], dtype=torch.float32))
        batch_x = torch.stack(imgs)
        batch_y = torch.stack(labels)
        if batch_transform is not None:
            batch_x, batch_y = batch_transform(batch_x, batch_y)

        y0 = r * (cell_h + label_h)
        cv2.putText(
            grid,
            variant,
            (8, y0 + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        for c in range(samples_per_variant):
            rgb = tensor_to_uint8_rgb(batch_x[c])
            x0 = variant_w + c * cell_w
            grid[y0 + label_h:y0 + label_h + cell_h, x0:x0 + cell_w] = rgb
            cv2.putText(
                grid,
                f"y={float(batch_y[c].item()):+.3f}",
                (x0 + 8, y0 + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_validation_loss_curves(history_rows: Sequence[Dict[str, Any]], output_path: str) -> bool:
    if not history_rows:
        return False

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in history_rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["epoch"]))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6), dpi=160)
        for variant, rows in grouped.items():
            epochs = [int(row["epoch"]) for row in rows]
            losses = [float(row["val_mse"]) for row in rows]
            plt.plot(epochs, losses, marker="o", linewidth=1.8, markersize=3.5, label=variant)
        plt.xlabel("Epoch")
        plt.ylabel("Validation MSE")
        plt.title("Validation Loss Curves")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True
    except Exception:
        return plot_validation_loss_curves_cv2(grouped, output_path)


def plot_validation_loss_curves_cv2(grouped: Dict[str, List[Dict[str, Any]]], output_path: str) -> bool:
    width, height = 1100, 700
    left, right, top, bottom = 90, 260, 70, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    all_epochs = [int(row["epoch"]) for rows in grouped.values() for row in rows]
    all_losses = [float(row["val_mse"]) for rows in grouped.values() for row in rows]
    if not all_epochs or not all_losses:
        return False
    min_epoch, max_epoch = min(all_epochs), max(all_epochs)
    min_loss, max_loss = 0.0, max(all_losses)
    if max_loss <= min_loss:
        max_loss = min_loss + 1.0

    def map_x(epoch: int) -> int:
        if max_epoch == min_epoch:
            return left + plot_w // 2
        return int(left + (epoch - min_epoch) / (max_epoch - min_epoch) * plot_w)

    def map_y(loss: float) -> int:
        return int(top + (max_loss - loss) / (max_loss - min_loss) * plot_h)

    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (30, 30, 30), 1)
    for i in range(6):
        y = top + int(i * plot_h / 5)
        loss = max_loss - i * (max_loss - min_loss) / 5
        cv2.line(canvas, (left, y), (left + plot_w, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"{loss:.4f}", (12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1, cv2.LINE_AA)
    for i in range(6):
        x = left + int(i * plot_w / 5)
        epoch = min_epoch if max_epoch == min_epoch else min_epoch + i * (max_epoch - min_epoch) / 5
        cv2.line(canvas, (x, top), (x, top + plot_h), (235, 235, 235), 1)
        cv2.putText(canvas, f"{epoch:.0f}", (x - 10, top + plot_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Validation Loss Curves", (left, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Epoch", (left + plot_w // 2 - 25, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Validation MSE", (12, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 55, 55), 1, cv2.LINE_AA)

    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
    ]
    for idx, (variant, rows) in enumerate(grouped.items()):
        color = colors[idx % len(colors)]
        points = [(map_x(int(row["epoch"])), map_y(float(row["val_mse"]))) for row in rows]
        for p0, p1 in zip(points[:-1], points[1:]):
            cv2.line(canvas, p0, p1, color, 2)
        for point in points:
            cv2.circle(canvas, point, 4, color, -1)
        legend_y = top + idx * 26
        legend_x = left + plot_w + 35
        cv2.line(canvas, (legend_x, legend_y), (legend_x + 28, legend_y), color, 3)
        cv2.putText(canvas, variant, (legend_x + 38, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)

    return bool(cv2.imwrite(output_path, canvas))


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: str, summaries: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> None:
    lines = [
        "# DonkeyCar Augmentation Ablation",
        "",
        "Task: steering-angle regression. Validation metrics are MSE, RMSE, and MAE.",
        "",
        f"- Device: {config['device']}",
        f"- Epochs: {config['epochs']}",
        f"- Train samples: {config['max_train_samples'] or 'all'}",
        f"- Val samples: {config['max_val_samples'] or 'all'}",
        f"- Train list: `{config['train_list']}`",
        f"- Val list: `{config['val_list']}`",
        f"- Batch size: {config['batch_size']}",
        f"- Transform source: `mytorch.transforms`",
        f"- Crop steering gain: {config['crop_steering_gain']}",
        f"- Rotation steering gain: {config['rotate_steering_gain']}",
        f"- Random crop enabled: {config['use_random_crop']}",
        f"- Random rotation enabled: {config['use_random_rotation']}",
        f"- Center crop height ratio: {config['center_crop_height_ratio']}",
        f"- Center crop width ratio: {config['center_crop_width_ratio']}",
        f"- Local MixUp max label diff: {config['local_mix_max_diff']}",
        "",
        "## Validation Metrics",
        "",
        "| variant | best epoch | best val MSE | best val RMSE | best val MAE | final val MSE | time sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not config["use_random_crop"] and not config["use_random_rotation"]:
        lines[14:14] = [
            "> Note: random crop and random rotation are disabled by default. "
            "`crop_rotate_jitter` currently means horizontal flip + color jitter + noise.",
            "",
        ]
    elif config["crop_steering_gain"] == 0.0 and config["rotate_steering_gain"] == 0.0:
        lines[14:14] = [
            "> Warning: crop/rotation steering compensation is disabled. "
            "`crop_rotate_jitter` is an explicitly uncalibrated spatial baseline, "
            "not a physically consistent augmentation policy.",
            "",
        ]
    for row in sorted(summaries, key=lambda r: r["best_val_mse"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    fmt(row["best_epoch"]),
                    fmt(row["best_val_mse"]),
                    fmt(row["best_val_rmse"]),
                    fmt(row["best_val_mae"]),
                    fmt(row["final_val_mse"]),
                    fmt(row["total_time_sec"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Augmentation Groups",
            "",
            "- `none`: resize + normalize only.",
            "- `center_crop`: fixed center crop then resize back to the original input size; by default it keeps full width and crops vertical context only.",
            "- `flip_only`: horizontal flip with steering sign inversion only. Use it to verify label direction.",
            "- `brightness_only`: brightness jitter only, using +/-10%.",
            "- `contrast_only`: contrast jitter only, using +/-10%.",
            "- `gamma_only`: gamma/exposure jitter only, using 0.9-1.1.",
            "- `noise_only`: Gaussian noise only, using std=0.005.",
            "- `hsv_only`: HSV saturation/value jitter only, using +/-10%.",
            "- `mixup_only`: batch-level MixUp only; no image-space augmentation before mixing.",
            "- `local_mixup_only`: MixUp only between nearby steering labels; no image-space augmentation before mixing.",
            "- `cutmix_only`: CutMix only; no image-space augmentation before patch replacement.",
            "- Legacy combined groups remain available if explicitly requested: `donkey_safe`, `local_mixup`, `basic`, `crop_rotate_jitter`, `mixup`, `cutmix`.",
            "",
            "## Transform Definitions",
            "",
            "```json",
            json.dumps(config["augmentation_definitions"], indent=2, ensure_ascii=False),
            "```",
            "",
            "Sample image grid: `augmentation_samples.png`.",
            "Validation loss curves: `val_loss_curves.png`.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="DonkeyCar data augmentation ablation.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--val-list", default="val.txt")
    parser.add_argument("--results-dir", default=os.path.join("results", "augmentation_ablation"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--mix-alpha", type=float, default=0.4)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--local-mix-alpha", type=float, default=0.2)
    parser.add_argument("--local-mix-max-diff", type=float, default=0.05)
    parser.add_argument(
        "--crop-steering-gain",
        type=float,
        default=0.0,
        help="Steering-label correction per normalized horizontal crop shift. Keep 0 until calibrated.",
    )
    parser.add_argument(
        "--rotate-steering-gain",
        type=float,
        default=0.0,
        help="Steering-label correction per image rotation degree. Keep 0 until calibrated.",
    )
    parser.add_argument(
        "--use-random-crop",
        action="store_true",
        help="Enable random crop in crop_rotate_jitter. Disabled by default until steering compensation is calibrated.",
    )
    parser.add_argument(
        "--use-random-rotation",
        action="store_true",
        help="Enable random rotation in crop_rotate_jitter. Disabled by default until steering compensation is calibrated.",
    )
    parser.add_argument(
        "--center-crop-height-ratio",
        type=float,
        default=0.85,
        help="Height ratio kept by center_crop before resizing back. Default keeps 85% of image height.",
    )
    parser.add_argument(
        "--center-crop-width-ratio",
        type=float,
        default=1.0,
        help="Width ratio kept by center_crop before resizing back. Default keeps full width to avoid steering-center shift.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "none",
            "center_crop",
            "flip_only",
            "brightness_only",
            "contrast_only",
            "gamma_only",
            "noise_only",
            "hsv_only",
            "mixup_only",
            "local_mixup_only",
            "cutmix_only",
        ],
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    sample_path = os.path.join(output_dir, "augmentation_samples.png")
    make_sample_grid(
        args.data_root,
        args.train_list,
        sample_path,
        seed=args.seed,
        crop_steering_gain=args.crop_steering_gain,
        rotate_steering_gain=args.rotate_steering_gain,
        use_random_crop=args.use_random_crop,
        use_random_rotation=args.use_random_rotation,
        center_crop_height_ratio=args.center_crop_height_ratio,
        center_crop_width_ratio=args.center_crop_width_ratio,
        mix_alpha=args.mix_alpha,
        cutmix_alpha=args.cutmix_alpha,
        local_mix_alpha=args.local_mix_alpha,
        local_mix_max_diff=args.local_mix_max_diff,
        variants=args.variants,
    )

    summaries: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    for idx, variant in enumerate(args.variants):
        set_seed(args.seed + idx)
        summary, history = train_variant(variant, args, device, output_dir)
        summaries.append(summary)
        history_rows.extend(history)

    val_loss_curve_path = os.path.join(output_dir, "val_loss_curves.png")
    plot_validation_loss_curves(history_rows, val_loss_curve_path)

    config = {
        **vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "output_dir": output_dir,
        "sample_grid": sample_path,
        "val_loss_curve": val_loss_curve_path,
        "train_list": resolve_list_path(args.data_root, args.train_list),
        "val_list": resolve_list_path(args.data_root, args.val_list),
        "augmentation_definitions": {
            variant: describe_donkeycar_augmentation(
                variant,
                crop_steering_gain=args.crop_steering_gain,
                rotate_steering_gain=args.rotate_steering_gain,
                mix_alpha=args.mix_alpha,
                cutmix_alpha=args.cutmix_alpha,
                local_mix_alpha=args.local_mix_alpha,
                local_mix_max_diff=args.local_mix_max_diff,
                use_random_crop=args.use_random_crop,
                use_random_rotation=args.use_random_rotation,
                center_crop_height_ratio=args.center_crop_height_ratio,
                center_crop_width_ratio=args.center_crop_width_ratio,
            )
            for variant in args.variants
        },
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"config": config, "summaries": summaries}, f, indent=2, ensure_ascii=False)
    write_csv(os.path.join(output_dir, "summary.csv"), summaries)
    write_csv(os.path.join(output_dir, "history.csv"), history_rows)
    write_markdown(os.path.join(output_dir, "summary.md"), summaries, config)

    print(f"Saved ablation results to: {output_dir}")
    print(os.path.join(output_dir, "summary.md"))
    print(sample_path)
    print(val_loss_curve_path)


if __name__ == "__main__":
    main()
