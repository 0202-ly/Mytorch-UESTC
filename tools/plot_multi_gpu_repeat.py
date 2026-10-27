"""Plot measured multi-GPU PS/Ring speed, convergence, and communication."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ORDER = ("socket2", "grpc2", "ring2", "socket4", "grpc4", "ring4")
LABELS = {"socket2": "Socket 2", "socket4": "Socket 4",
          "grpc2": "gRPC 2", "grpc4": "gRPC 4",
          "ring2": "Ring 2", "ring4": "Ring 4"}
COLORS = {"socket2": "#4C78A8", "socket4": "#4C78A8",
          "grpc2": "#F58518", "grpc4": "#F58518",
          "ring2": "#54A24B", "ring4": "#54A24B"}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {row["configuration"]: row for row in read_rows(args.aggregate_dir / "multi_gpu_summary.csv")}
    if set(summary) != set(ORDER):
        raise ValueError("all six multi-GPU configurations are required")

    measured = [float(summary[mode]["speedup_vs_single"]) for mode in ORDER]
    ideal = [float(summary[mode]["ideal_speedup"]) for mode in ORDER]
    x = list(range(len(ORDER)))
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=170)
    ax.bar([i - 0.2 for i in x], measured, width=0.38,
           color=[COLORS[mode] for mode in ORDER], label="Measured")
    ax.bar([i + 0.2 for i in x], ideal, width=0.38,
           color="#CAD2DC", label="Ideal linear")
    for i, value in enumerate(measured):
        ax.text(i - 0.2, value + 0.06, f"{value:.2f}x", ha="center", fontsize=8)
    ax.set_xticks(x, [LABELS[mode] for mode in ORDER])
    ax.set_ylabel("Speedup versus one GPU")
    ax.set_title("10-epoch measured and ideal speedup: one GPU per worker")
    ax.set_ylim(0, max(ideal) * 1.12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "multi_gpu_speedup.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=170)
    for mode in ORDER:
        rows = read_rows(args.aggregate_dir / f"{mode}_aggregate.csv")
        epoch = [int(row["epoch"]) + 1 for row in rows]
        seconds = [float(row["total_ms"]) / 1000 for row in rows]
        accuracy = [float(row["accuracy"]) for row in rows]
        axes[0].plot(epoch, seconds, marker="o", ms=3, color=COLORS[mode],
                     linestyle="-" if mode.endswith("2") else "--", label=LABELS[mode])
        axes[1].plot(epoch, accuracy, marker="o", ms=3, color=COLORS[mode],
                     linestyle="-" if mode.endswith("2") else "--", label=LABELS[mode])
    axes[0].set(title="Epoch completion time", xlabel="Epoch", ylabel="Seconds")
    axes[1].set(title="Training threshold accuracy", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes:
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "multi_gpu_epochs_and_accuracy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=170)
    comm = [float(summary[mode]["avg_communication_pct"]) * 100 for mode in ORDER]
    ax.bar(x, comm, color=[COLORS[mode] for mode in ORDER])
    ax.set_xticks(x, [LABELS[mode] for mode in ORDER])
    ax.set(title="Share of epoch time spent on communication", ylabel="Communication share (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "multi_gpu_communication_share.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
