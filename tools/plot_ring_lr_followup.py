"""Compare Ring 4-worker training accuracy at base and linearly scaled LR."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_mean_accuracy(folder: Path, workers: int = 4) -> list[float]:
    per_worker = []
    for rank in range(workers):
        with (folder / f"ring_worker{rank}.csv").open(encoding="utf-8-sig", newline="") as handle:
            per_worker.append(list(csv.DictReader(handle)))
    if any(len(rows) != 10 for rows in per_worker):
        raise ValueError(f"{folder}: expected 10 epochs for each worker")
    return [sum(float(rows[epoch]["accuracy"]) for rows in per_worker) / workers
            for epoch in range(10)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--scaled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = load_mean_accuracy(args.original_dir)
    scaled = load_mean_accuracy(args.scaled_dir)
    with (args.output_dir / "ring4_learning_rate_accuracy.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "lr_0_001_accuracy", "lr_0_004_accuracy"))
        writer.writerows((epoch + 1, original[epoch], scaled[epoch]) for epoch in range(10))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(range(1, 11), original, "o-", label="Ring 4 GPU, LR=0.001")
    ax.plot(range(1, 11), scaled, "o-", label="Ring 4 GPU, LR=0.004")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training threshold accuracy (|steering error| < 0.05)")
    ax.set_title("Ring 4 GPU convergence follow-up: learning-rate scaling")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "ring4_learning_rate_accuracy.png")
    plt.close(fig)
    print(f"final accuracy: LR 0.001={original[-1]:.4f}, LR 0.004={scaled[-1]:.4f}")


if __name__ == "__main__":
    main()
