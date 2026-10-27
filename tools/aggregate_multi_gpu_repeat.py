"""Validate and summarize complete 1/2/4 GPU training runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODES = ("socket2", "socket4", "grpc2", "grpc4", "ring2", "ring4")
FIELDS = ("backend", "workers", "epoch", "total_ms", "compute_ms", "communication_ms", "message_bytes", "accuracy")


def rows_from(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = rows_from(args.baseline)
    if [int(row["epoch"]) for row in baseline_rows] != list(range(10)):
        raise ValueError("single-GPU baseline must contain epochs 0 through 9")
    baseline_ms = sum(float(row["total_ms"]) for row in baseline_rows)
    write_csv(args.output_dir / "single_aggregate.csv", baseline_rows, FIELDS)
    summaries = []
    for mode in MODES:
        folder = args.run_dir / mode
        metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
        workers = int(mode[-1])
        if metadata["epochs"] != 10 or metadata["workers"] != workers or metadata["gpu_ids"] != list(range(workers)):
            raise ValueError(f"{mode}: metadata does not match the 10-epoch dedicated GPU experiment")
        prefix = "ring_worker" if mode.startswith("ring") else "worker"
        by_worker = [rows_from(folder / f"{prefix}{rank}.csv") for rank in range(workers)]
        for rank, rows in enumerate(by_worker):
            if [int(row["epoch"]) for row in rows] != list(range(10)):
                raise ValueError(f"{mode} worker {rank}: missing or duplicate epoch")
            if any(int(row["workers"]) != workers or int(row["message_bytes"]) <= 0 for row in rows):
                raise ValueError(f"{mode} worker {rank}: inconsistent worker count or missing communication")
        monitor = rows_from(folder / "gpu_monitor.csv")
        monitored = {int(row["gpu_index"]) for row in monitor if float(row["memory_used_mib"]) > 1000}
        if not set(range(workers)).issubset(monitored):
            raise ValueError(f"{mode}: GPU monitoring does not confirm all assigned cards: {monitored}")
        assigned_samples = [row for row in monitor if int(row["gpu_index"]) < workers]
        aggregated = []
        for epoch in range(10):
            items = [rows[epoch] for rows in by_worker]
            aggregated.append({
                "backend": mode,
                "workers": workers,
                "epoch": epoch,
                "total_ms": max(float(row["total_ms"]) for row in items),
                "compute_ms": sum(float(row["compute_ms"]) for row in items) / workers,
                "communication_ms": sum(float(row["communication_ms"]) for row in items) / workers,
                "message_bytes": sum(int(row["message_bytes"]) for row in items) / workers,
                "accuracy": sum(float(row["accuracy"]) for row in items) / workers,
            })
        write_csv(args.output_dir / f"{mode}_aggregate.csv", aggregated, FIELDS)
        total_ms = sum(row["total_ms"] for row in aggregated)
        summaries.append({
            "configuration": mode,
            "workers": workers,
            "gpu_count": workers,
            "epochs": 10,
            "train_samples": 7860,
            "total_seconds": total_ms / 1000,
            "launcher_wall_seconds": metadata["wall_seconds"],
            "speedup_vs_single": baseline_ms / total_ms,
            "ideal_speedup": workers,
            "final_accuracy": aggregated[-1]["accuracy"],
            "avg_communication_pct": sum(row["communication_ms"] / row["total_ms"] for row in aggregated) / 10,
            "mean_message_bytes_per_worker_epoch": sum(row["message_bytes"] for row in aggregated) / 10,
            "mean_gpu_utilization_pct": sum(float(row["utilization_pct"]) for row in assigned_samples) / len(assigned_samples),
            "peak_gpu_memory_mib": max(float(row["memory_used_mib"]) for row in assigned_samples),
        })
    columns = tuple(summaries[0].keys())
    write_csv(args.output_dir / "multi_gpu_summary.csv", summaries, columns)
    (args.output_dir / "multi_gpu_summary.json").write_text(
        json.dumps({"baseline_total_seconds": baseline_ms / 1000, "runs": summaries}, indent=2),
        encoding="utf-8",
    )
    print((args.output_dir / "multi_gpu_summary.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    main()
