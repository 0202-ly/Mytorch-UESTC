"""Make presentation-ready plots from distributed training CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for raw in args.files:
        path = Path(raw)
        rows = read_rows(path)
        if not rows:
            continue
        groups[path.stem] = rows

    # Epoch-level convergence curves.
    plt.figure(figsize=(8.5, 5.0), dpi=160)
    for name, rows in groups.items():
        epochs = [int(r["epoch"]) + 1 for r in rows]
        accuracy = [float(r["accuracy"]) for r in rows]
        if len(epochs) > 1:
            plt.plot(epochs, accuracy, marker="o", linewidth=1.8, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Threshold accuracy")
    plt.title("Distributed training convergence")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "distributed_convergence.png")
    plt.close()

    # Measured speedup versus the ideal linear curve.
    baseline = None
    for name, rows in groups.items():
        if "single" in name.lower():
            baseline = sum(float(r["total_ms"]) for r in rows)
            break
    if baseline is not None:
        labels, measured, ideal = [], [], []
        for name, rows in groups.items():
            if "single" in name.lower():
                continue
            workers = int(rows[0].get("workers", 1))
            total = sum(float(r["total_ms"]) for r in rows)
            labels.append(name.replace("_aggregate", "").replace("_acc4_fp16", ""))
            measured.append(baseline / total)
            ideal.append(float(workers))
        x = list(range(len(labels)))
        plt.figure(figsize=(9.0, 5.0), dpi=160)
        width = 0.36
        plt.bar([v - width / 2 for v in x], measured, width, label="Measured")
        plt.bar([v + width / 2 for v in x], ideal, width, label="Ideal linear")
        plt.xticks(x, labels, rotation=20, ha="right", fontsize=8)
        plt.ylabel("Speedup")
        plt.title("Measured speedup versus ideal linear speedup")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        for idx, value in enumerate(measured):
            plt.text(idx - width / 2, value + 0.03, f"{value:.2f}", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "speedup_measured_vs_linear.png")
        plt.close()

    with (out / "optimization_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "epoch", "total_ms", "compute_ms", "communication_ms", "message_bytes", "accuracy"])
        for name, rows in groups.items():
            for row in rows:
                writer.writerow([name, row["epoch"], row["total_ms"], row["compute_ms"], row["communication_ms"], row["message_bytes"], row["accuracy"]])


if __name__ == "__main__":
    main()
