from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


FIELDS = ["backend", "workers", "epoch", "total_ms", "compute_ms", "communication_ms", "message_bytes", "accuracy"]


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def aggregate(paths, output, backend_name=None):
    grouped = defaultdict(list)
    for path in paths:
        for row in read(path):
            grouped[int(row["epoch"])].append(row)
    rows = []
    for epoch in sorted(grouped):
        items = grouped[epoch]
        rows.append({
            "backend": backend_name or items[0]["backend"],
            "workers": items[0]["workers"],
            "epoch": epoch,
            "total_ms": max(float(x["total_ms"]) for x in items),
            "compute_ms": sum(float(x["compute_ms"]) for x in items) / len(items),
            "communication_ms": sum(float(x["communication_ms"]) for x in items) / len(items),
            "message_bytes": sum(int(x["message_bytes"]) for x in items) / len(items),
            "accuracy": sum(float(x["accuracy"]) for x in items) / len(items),
        })
    with Path(output).open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output-dir", required=True); p.add_argument("--single", required=True); p.add_argument("--socket2", nargs="+", required=True); p.add_argument("--socket4", nargs="+", required=True); p.add_argument("--ring2", nargs="+", required=True)
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    groups = {
        "single_true10": aggregate([a.single], out / "single_true10_aggregate.csv", "single_true10"),
        "socket2_acc4_fp16": aggregate(a.socket2, out / "socket2_acc4_fp16_aggregate.csv", "socket2_acc4_fp16"),
        "socket4_acc4_fp16": aggregate(a.socket4, out / "socket4_acc4_fp16_aggregate.csv", "socket4_acc4_fp16"),
        "ring2_acc4_fp16": aggregate(a.ring2, out / "ring2_acc4_fp16_aggregate.csv", "ring2_acc4_fp16"),
    }
    baseline = sum(float(x["total_ms"]) for x in groups["single_true10"])
    summary = {"baseline_total_ms": baseline, "groups": {}}
    for name, rows in groups.items():
        total = sum(float(x["total_ms"]) for x in rows)
        summary["groups"][name] = {"epochs": len(rows), "total_ms": total, "speedup_vs_single": baseline / total if total else None, "final_accuracy": rows[-1]["accuracy"] if rows else None, "avg_communication_pct": sum(float(x["communication_ms"]) / max(float(x["total_ms"]), 1e-9) for x in rows) / max(len(rows), 1)}
    (out / "distributed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
