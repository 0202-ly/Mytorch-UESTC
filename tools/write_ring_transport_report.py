"""Write the four requested PS-vs-Ring transport comparisons from validated CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads((args.transport_run_dir / "aggregate" / "all_transport_summary.json").read_text(encoding="utf-8"))
    rows = {row["configuration"]: row for row in payload["runs"]}
    pairs = (("socket2", "ring_socket2"), ("socket4", "ring_socket4"),
             ("grpc2", "ring_grpc2"), ("grpc4", "ring_grpc4"))
    lines = [
        "# Ring AllReduce：Socket+JSON 与 gRPC+Protobuf 四组实测",
        "",
        "四组均使用当前项目 train.txt 可读取的全部 7860 张 DonkeyCar 图片（不能据此证明它是此前采集的原始全集）、10 epoch、每 Worker 专用 GPU、batch 16、累积 4 个 batch 后同步、FP16 梯度。PS 与 Ring 使用相同初始权重；Socket+JSON Ring 用 JSON 封装 base64 FP16 梯度块，gRPC+Protobuf Ring 用 Protobuf bytes 携带 FP16 梯度块。前者不是把 FP16 逐元素写成十进制 JSON 数组。",
        "",
        "| 同一种通信实现 | PS 10 epoch (s) | Ring 10 epoch (s) | Ring/PS 时间比 | Ring 相对单卡加速比 | Ring 末轮阈值准确率 | Ring 平均单 Worker 每轮收发 (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ps, ring in pairs:
        a, b = rows[ps], rows[ring]
        label = ("Socket+JSON" if ps.startswith("socket") else "gRPC+Protobuf") + f"，{a['workers']} Worker"
        lines.append(f"| {label} | {float(a['total_seconds']):.1f} | {float(b['total_seconds']):.1f} | {float(b['total_seconds'])/float(a['total_seconds']):.3f} | {float(b['speedup_vs_single']):.3f}× | {float(b['final_accuracy']):.4f} | {float(b['mean_message_bytes_per_worker_epoch'])/1e9:.3f} |")
    lines += [
        "",
        "此处的 `Ring/PS 时间比` 小于 1 表示 Ring 更快。PS 异步更新、Ring 同步更新，有效 batch 和更新步数不同，故速度对照**不是同等模型精度下的提速**。Socket PS 使用十进制 JSON 数组，Socket Ring 使用 JSON+base64，编码也不完全相同，因此此表不能独立归因于架构差异。要宣称收敛，还需确认准确率或损失在后期趋稳；不能仅凭跑满 10 epoch。",
        "",
        "[十组汇总原始 CSV](results/ring_transport_full_v1/aggregate/all_transport_summary.csv) · [实测与理想加速比曲线](results/ring_transport_full_v1/aggregate/all_transport_speedup_curve.png) · [三种 Ring 传输的数值正确性日志](results/ring_transport_math_validation.log) · [四组训练启动器日志](results/ring_transport_full_v1_launcher.log)",
        "",
    ]
    for _, ring in pairs:
        lines.append(f"- {ring}：[逐轮 CSV](results/ring_transport_full_v1/{ring}/epoch_aggregate.csv)、[每 Worker 原始记录](results/ring_transport_full_v1/{ring}/ring_worker0.csv)、[GPU 监控](results/ring_transport_full_v1/{ring}/gpu_monitor.csv)、[终端日志](results/ring_transport_full_v1/{ring}/ring.log)。")
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
