"""Write a reproducible report from the validated six-mode multi-GPU summary."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABEL = {
    "socket2": "Socket+JSON，2 Worker / 2 GPU",
    "socket4": "Socket+JSON，4 Worker / 4 GPU",
    "grpc2": "gRPC+Protobuf，2 Worker / 2 GPU",
    "grpc4": "gRPC+Protobuf，4 Worker / 4 GPU",
    "ring2": "Ring AllReduce，2 Worker / 2 GPU",
    "ring4": "Ring AllReduce，4 Worker / 4 GPU",
}
ORDER = ("socket2", "grpc2", "ring2", "socket4", "grpc4", "ring4")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    summary = json.loads((root / "aggregate" / "multi_gpu_summary.json").read_text(encoding="utf-8"))
    rows = {row["configuration"]: row for row in summary["runs"]}
    if set(rows) != set(ORDER):
        raise ValueError("expected all six validated configurations")
    for mode in ORDER:
        if len(read_rows(root / "aggregate" / f"{mode}_aggregate.csv")) != 10:
            raise ValueError(f"{mode}: expected 10 epochs")

    lines = [
        "# 114 号机：独立 GPU 的分布式 CNN 训练实测",
        "",
        "## 实验条件",
        "",
        "- 4 块 NVIDIA GeForce RTX 4090 D；每个 Worker 独占一块 GPU，PS 运行在 CPU。",
        "- 使用当前项目 train.txt 可读取的全部 7860 张 DonkeyCar 图片（缺失 0 张）；无法仅凭当前仓库证明它涵盖了此前采集的所有原始数据。每组训练 10 epoch，batch size 16/Worker，学习率 0.001。",
        "- 分布式版本每累积 4 个 batch 同步一次梯度，梯度传输使用 FP16；六组使用同一模型初始权重。",
        "- 单 GPU 基线也使用相同初始权重和 10 epoch，batch size 16。各组处理相同数量的图片，但同步方式和有效 batch 不完全相同。",
        "- 准确率是当前项目的转向角阈值准确率（绝对误差 < 0.05），不是分类准确率。",
        "",
        "## 实测结果",
        "",
        f"单 GPU 10 epoch 基线：**{summary['baseline_total_seconds']:.1f} 秒**。实测加速比 = 单 GPU 总时间 / 多 GPU 总时间；理想线性加速比分别为 2 或 4。",
        "",
        "| 配置 | 10 epoch 时间 (s) | 实测加速比 | 理想值 | 与理想值差距 | 末 epoch 准确率 | 平均通信占比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ORDER:
        row = rows[mode]
        measured = row["speedup_vs_single"]
        ideal = row["ideal_speedup"]
        lines.append(f"| {LABEL[mode]} | {row['total_seconds']:.1f} | {measured:.3f}× | {ideal:.0f}× | {ideal-measured:.3f}× | {row['final_accuracy']:.3f} | {row['avg_communication_pct']*100:.1f}% |")

    lines += [
        "",
        "## PS 与 Ring 的四组对照",
        "",
        "| 对照 | PS 时间 (s) | Ring 时间 (s) | Ring 相对 PS 提速 |",
        "|---|---:|---:|---:|",
    ]
    for ps, ring in (("socket2", "ring2"), ("socket4", "ring4"), ("grpc2", "ring2"), ("grpc4", "ring4")):
        ps_time = rows[ps]["total_seconds"]
        ring_time = rows[ring]["total_seconds"]
        lines.append(f"| {LABEL[ps]} ↔ {LABEL[ring]} | {ps_time:.1f} | {ring_time:.1f} | {ps_time/ring_time:.3f}× |")

    fast = [mode for mode in ORDER if rows[mode]["speedup_vs_single"] > 1]
    slow = [mode for mode in ORDER if rows[mode]["speedup_vs_single"] <= 1]
    lines += [
        "",
        "## 观察与边界",
        "",
        f"- 比单 GPU 更快的组：{', '.join(LABEL[mode] for mode in fast) if fast else '无'}。",
        f"- 未超过单 GPU 的组：{', '.join(LABEL[mode] for mode in slow) if slow else '无'}。",
        "- 加速比按每个 epoch 最慢 Worker 的完成时间求和；启动器还单独记录了含进程启动的实际墙钟时间。",
        "- PS 的消息字节数记录梯度请求与参数返回的主要载荷；Ring 记录发送与接收的梯度块，因此协议间字节数不能直接视作完整网络总流量。",
        "- 10 epoch 的完成不自动证明模型已收敛。应结合逐 epoch 准确率曲线判断；不同同步方式可能改变有效 batch 和最终精度。",
        "- 尤其 Ring 4 Worker 最后一轮准确率约 0.25，且末轮未持续上升，因此本次不能宣称其已收敛。Ring 每次同步合并所有 Worker 的梯度，PS 则按 Worker 上传先后异步更新；相同 epoch 数下，二者参数更新次数和有效 batch 不同。该表是训练速度对照，不是同等精度达标时间对照。",
        "- Ring 2/4 这两组使用 FP16 二进制 TCP；Socket+JSON 与 gRPC+Protobuf 的 Ring 传输变体是另行补跑的四组，未混入此表。",
        "- 本次是单机多 GPU 实验，无法据此推断跨机器网络上的扩展性能。",
        "",
        "## 原始证据与图表",
        "",
        "- [完整汇总 CSV](results/multi_gpu_full_v1/aggregate/multi_gpu_summary.csv)、[实测与线性加速比图](results/multi_gpu_full_v1/aggregate/multi_gpu_speedup.png)、[epoch 时间与准确率曲线](results/multi_gpu_full_v1/aggregate/multi_gpu_epochs_and_accuracy.png)、[通信占比图](results/multi_gpu_full_v1/aggregate/multi_gpu_communication_share.png)。",
        "- [单 GPU 原始 CSV](results/multi_gpu_full_v1/single_gpu_same_initial.csv) 与 [单 GPU 终端日志](results/multi_gpu_full_v1/single_gpu_same_initial.log)。",
        "- [Ring 2/4 Worker 数值正确性日志](results/multi_gpu_full_v1/ring_math_validation.log)。",
        "- [DonkeyCar 训练集核对记录](results/multi_gpu_full_v1/dataset_audit.json)：7860 行、7860 个不同图片路径、缺失 0 张。",
    ]
    for mode in ORDER:
        prefix = "ring_worker0.csv" if mode.startswith("ring") else "worker0.csv"
        log = "ring.log" if mode.startswith("ring") else "worker0.log"
        lines.append(f"- {LABEL[mode]}：[Worker 0 CSV](results/multi_gpu_full_v1/{mode}/{prefix})、[GPU 监控 CSV](results/multi_gpu_full_v1/{mode}/gpu_monitor.csv)、[终端日志](results/multi_gpu_full_v1/{mode}/{log})。")
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
