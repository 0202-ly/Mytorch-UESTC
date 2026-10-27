"""Validate the four Ring transport runs and compare against PS and ideal speedup."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


TRANSPORT_MODES = ("ring_socket2", "ring_socket4", "ring_grpc2", "ring_grpc4")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_transport(root: Path, mode: str, baseline: float) -> dict:
    folder = root / mode
    workers = int(mode[-1])
    metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete" or metadata["epochs"] != 10:
        raise ValueError(f"{mode}: incomplete metadata")
    per_worker = [read_csv(folder / f"ring_worker{rank}.csv") for rank in range(workers)]
    if any(len(rows) != 10 or [int(row["epoch"]) for row in rows] != list(range(10))
           for rows in per_worker):
        raise ValueError(f"{mode}: expected 10 complete epochs for every worker")
    gpu_rows = read_csv(folder / "gpu_monitor.csv")
    for gpu_id in range(workers):
        if not any(int(row["gpu_index"]) == gpu_id and float(row["memory_used_mib"]) > 2000
                   for row in gpu_rows):
            raise ValueError(f"{mode}: GPU {gpu_id} was not observed as active")
    epoch_rows = []
    for epoch in range(10):
        values = [rows[epoch] for rows in per_worker]
        epoch_rows.append({
            "epoch": epoch + 1,
            "total_seconds": max(float(row["total_ms"]) for row in values) / 1000,
            "mean_compute_seconds": sum(float(row["compute_ms"]) for row in values) / workers / 1000,
            "mean_communication_seconds": sum(float(row["communication_ms"]) for row in values) / workers / 1000,
            "mean_message_bytes_per_worker": sum(int(row["message_bytes"]) for row in values) / workers,
            "mean_accuracy": sum(float(row["accuracy"]) for row in values) / workers,
        })
    save_csv(folder / "epoch_aggregate.csv", epoch_rows)
    total = sum(row["total_seconds"] for row in epoch_rows)
    return {
        "configuration": mode,
        "workers": workers,
        "gpu_count": workers,
        "epochs": 10,
        "total_seconds": total,
        "speedup_vs_single": baseline / total,
        "ideal_speedup": workers,
        "final_accuracy": epoch_rows[-1]["mean_accuracy"],
        "mean_communication_pct": sum(row["mean_communication_seconds"] / row["total_seconds"] for row in epoch_rows) / 10,
        "mean_message_bytes_per_worker_epoch": sum(row["mean_message_bytes_per_worker"] for row in epoch_rows) / 10,
        "launcher_wall_seconds": metadata["wall_seconds"],
    }


def plot_speedup(path: Path, baseline_rows: list[dict], transport_rows: list[dict]) -> None:
    rows = {row["configuration"]: row for row in baseline_rows + transport_rows}
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.plot([1, 2, 4], [1, 2, 4], "k--", linewidth=2, label="Ideal linear")
    curves = (
        ("Socket+JSON PS", ("socket2", "socket4"), "#4e79a7"),
        ("gRPC+Protobuf PS", ("grpc2", "grpc4"), "#f28e2b"),
        ("Binary TCP Ring", ("ring2", "ring4"), "#59a14f"),
        ("Socket+JSON Ring", ("ring_socket2", "ring_socket4"), "#b07aa1"),
        ("gRPC+Protobuf Ring", ("ring_grpc2", "ring_grpc4"), "#e15759"),
    )
    for name, keys, color in curves:
        y = [1.0] + [float(rows[key]["speedup_vs_single"]) for key in keys]
        ax.plot([1, 2, 4], y, "o-", linewidth=2, color=color, label=name)
        for x, value in zip((2, 4), y[1:]):
            ax.annotate(f"{value:.2f}x", (x, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("Workers / dedicated GPUs")
    ax.set_ylabel("10-epoch speedup versus 1 GPU")
    ax.set_title("Measured versus ideal distributed training speedup")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--transport-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.base_run_dir / "aggregate" / "multi_gpu_summary.json").read_text(encoding="utf-8"))
    baseline = float(base["baseline_total_seconds"])
    new_rows = [aggregate_transport(args.transport_run_dir, mode, baseline) for mode in TRANSPORT_MODES]
    combined = [
        {key: (row.get("avg_communication_pct", "") if key == "mean_communication_pct" else row.get(key, ""))
         for key in new_rows[0]}
        for row in base["runs"]
    ] + new_rows
    save_csv(args.output_dir / "all_transport_summary.csv", combined)
    (args.output_dir / "all_transport_summary.json").write_text(
        json.dumps({"baseline_total_seconds": baseline, "runs": combined}, indent=2), encoding="utf-8")
    plot_speedup(args.output_dir / "all_transport_speedup_curve.png", base["runs"], new_rows)
    for row in new_rows:
        print(f"{row['configuration']}: {row['total_seconds']:.2f}s; speedup={row['speedup_vs_single']:.3f}; accuracy={row['final_accuracy']:.3f}")


if __name__ == "__main__":
    main()
