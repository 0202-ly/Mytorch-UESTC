"""Render GPU/CPU resources and synchronized MyTorch step timing."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    args = parser.parse_args()
    steps = read(args.profile_dir / "step_metrics.csv")
    resources = read(args.profile_dir / "gpu_cpu_resources.csv")
    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=170)
    bottom = np.zeros(len(steps))
    for column, label, color in (
        ("data_prepare_ms", "Load + transfer", "#4C78A8"),
        ("forward_ms", "Forward", "#F58518"),
        ("backward_ms", "Backward", "#54A24B"),
        ("optimizer_update_ms", "Optimizer update", "#E45756"),
    ):
        values = np.asarray([float(row[column]) for row in steps])
        ax.bar(x + 1, values, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set(xlabel="Training step", ylabel="Synchronized time (ms)",
           title="MyTorch CNN training step breakdown")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.profile_dir / "step_timing_breakdown.png")
    plt.close(fig)

    t = np.arange(len(resources))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=170, sharex=True)
    axes[0].plot(t, [float(row["gpu_util_pct"]) for row in resources], marker="o", label="GPU %")
    axes[0].plot(t, [float(row["cpu_system_util_pct"]) for row in resources], marker="o", label="System CPU %")
    axes[0].plot(t, [float(row["cpu_process_util_pct_one_core"]) for row in resources], marker="o", label="Training process CPU % (one core=100)")
    axes[0].set_ylabel("Utilization (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, [float(row["memory_used_mib"]) for row in resources],
                 marker="o", color="#4C78A8", label="VRAM MiB")
    power_axis = axes[1].twinx()
    power_axis.plot(t, [float(row["power_w"]) for row in resources],
                    marker="o", color="#F58518", label="GPU power W")
    axes[1].set(xlabel="Sample", ylabel="VRAM (MiB)")
    power_axis.set_ylabel("Power (W)")
    axes[1].legend(loc="upper left", fontsize=8)
    power_axis.legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.suptitle("GPU 3 and CPU sampling during 20 MyTorch steps")
    fig.tight_layout()
    fig.savefig(args.profile_dir / "gpu_cpu_resource_monitor.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
