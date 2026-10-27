"""Create the two course-requested communication and measured-training tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--transport-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.base_run_dir
    summary = json.loads((root / "aggregate" / "multi_gpu_summary.json").read_text(encoding="utf-8"))
    by_mode = {row["configuration"]: row for row in summary["runs"]}
    theoretical = []
    for n in (2, 4, 8, 16):
        theoretical.append({
            "nodes_n": n,
            "ps_center_total_in_units_of_m": 2 * (n - 1),
            "ring_each_node_in_units_of_m": 2 * (n - 1) / n,
            "ps_center_over_ring_each_node": n,
        })
    write_rows(args.output_dir / "communication_volume_theory.csv", theoretical)

    modes = (
        ("single", "单机（基线）"),
        ("socket2", "Socket+JSON PS，2 Worker / 2 GPU"),
        ("grpc2", "gRPC+Protobuf PS，2 Worker / 2 GPU"),
        ("socket4", "Socket+JSON PS，4 Worker / 4 GPU"),
        ("grpc4", "gRPC+Protobuf PS，4 Worker / 4 GPU"),
        ("ring2", "二进制 TCP Ring AllReduce，2 Worker / 2 GPU"),
        ("ring4", "二进制 TCP Ring AllReduce，4 Worker / 4 GPU"),
    )
    measured = []
    for mode, label in modes:
        rows = read_rows(root / "aggregate" / f"{mode}_aggregate.csv")
        if len(rows) != 10:
            raise ValueError(f"{mode}: expected 10 epochs")
        total = sum(float(row["total_ms"]) for row in rows) / 10 / 1000
        compute = sum(float(row["compute_ms"]) for row in rows) / 10 / 1000
        communication = sum(float(row["communication_ms"]) for row in rows) / 10 / 1000
        speedup = 1.0 if mode == "single" else float(by_mode[mode]["speedup_vs_single"])
        measured.append({
            "configuration": label,
            "mean_epoch_total_seconds": round(total, 3),
            "mean_epoch_compute_seconds": round(compute, 3),
            "mean_epoch_communication_seconds": round(communication, 3),
            "speedup_vs_single": round(speedup, 3),
            "last_epoch_threshold_accuracy": round(float(rows[-1]["accuracy"]), 4),
        })
    if args.transport_run_dir:
        transport_summary = json.loads((args.transport_run_dir / "aggregate" / "all_transport_summary.json").read_text(encoding="utf-8"))
        transport_by_mode = {row["configuration"]: row for row in transport_summary["runs"]}
        for mode, label in (
            ("ring_socket2", "Socket+JSON Ring，2 Worker / 2 GPU"),
            ("ring_socket4", "Socket+JSON Ring，4 Worker / 4 GPU"),
            ("ring_grpc2", "gRPC+Protobuf Ring，2 Worker / 2 GPU"),
            ("ring_grpc4", "gRPC+Protobuf Ring，4 Worker / 4 GPU"),
        ):
            rows = read_rows(args.transport_run_dir / mode / "epoch_aggregate.csv")
            if len(rows) != 10:
                raise ValueError(f"{mode}: expected 10 epochs")
            measured.append({
                "configuration": label,
                "mean_epoch_total_seconds": round(sum(float(row["total_seconds"]) for row in rows) / 10, 3),
                "mean_epoch_compute_seconds": round(sum(float(row["mean_compute_seconds"]) for row in rows) / 10, 3),
                "mean_epoch_communication_seconds": round(sum(float(row["mean_communication_seconds"]) for row in rows) / 10, 3),
                "speedup_vs_single": round(float(transport_by_mode[mode]["speedup_vs_single"]), 3),
                "last_epoch_threshold_accuracy": round(float(rows[-1]["mean_accuracy"]), 4),
            })
    write_rows(args.output_dir / "course_measured_epoch_table.csv", measured)

    lines = [
        "# 课程要求的两张表：通信量与训练实测",
        "",
        "## 表一：两种架构的理论通信量",
        "",
        "令 M 为一次同步的完整梯度/参数大小。为与课件数字一致，这张表中的 N 对 PS 指 **1 个 PS + (N−1) 个 Worker**，对 Ring 指 **N 个 Worker**；两者的 Worker 数并不相同，属于拓扑示意而非严格同负载实测。只计主要梯度/参数载荷，不计协议头。",
        "",
        "| 节点数 N | PS 中心节点总通信量 | Ring 单节点通信量 | PS 是 Ring 的几倍 |",
        "|---:|---:|---:|---:|",
    ]
    for row in theoretical:
        lines.append(f"| {row['nodes_n']} | {row['ps_center_total_in_units_of_m']:g}M | {row['ring_each_node_in_units_of_m']:g}M | {row['ps_center_over_ring_each_node']} 倍 |")
    lines += [
        "",
        "PS：每个 Worker 上传 M 梯度、下载 M 参数，中心节点合计收发 2(N−1)M，随节点数线性增长。",
        "",
        "Ring 三步：**Split** 把 M 分为 N 份（本地切分、无网络流量）；**ScatterReduce（标准术语 Reduce-Scatter）** 用 N−1 轮让各节点持有一个求和后的分块，每节点发送约 (N−1)M/N；**AllGather（标准术语 All-Gather）** 再用 N−1 轮传播所有分块，每节点再发送约 (N−1)M/N。于是每节点总发送量 2(N−1)M/N，N 增大时趋近 2M，并非严格与 N 完全无关。",
        "",
        "注意：Ring 的**全网总发送量**仍为 2(N−1)M，随 N 增长；优势是流量分摊到节点，没有 PS 中心热点。课件表格还混用了方向口径：PS 的 2(N−1)M 是**发送+接收**，Ring 的 2(N−1)M/N 是**仅发送**；若两边都计发送+接收，Ring 应为 4(N−1)M/N，课件最后一列的倍率应减半为 N/2。这里保留课件原数值供汇报对照，但不能把它当严格同口径性能比。若公平地固定 **N 个 Worker**，PS 还要额外一个 PS 节点，其中心收发是 2NM。FP16、JSON 封装、Protobuf 和协议头也会改变实际字节数。",
        "",
        "如果链路带宽为 B、每轮启动延迟为 α，Ring 单节点通信时间的粗略模型是 2(N−1)α + 2(N−1)M/(NB)。虽然字节量接近常数，轮次数随 N 增长，小模型或高延迟网络下 Ring 也未必快。",
        "",
        "本项目的模型有 11,181,825 个参数：FP32 参数为 44,727,300 字节，FP16 梯度为 22,363,650 字节。若仅代入 FP16 梯度 M，Ring 每次同步的**单 Worker 发送量**理论约为 22.36 MB（2 Worker）或 33.55 MB（4 Worker）；原始 CSV 的 `message_bytes` 统计发送**加接收**，因此约是这些数的两倍再乘同步次数。PS 实际上传 FP16 梯度、下发 FP32 参数，Socket+JSON 还有封装开销，不能直接把实测字节与课件等精度的 M 公式相等。",
        "",
        "实际固定 7860 张图/epoch 时，Socket PS 平均每 Worker 每轮主要载荷约为 4.622 GB（2 Worker）或 2.311 GB（4 Worker）；乘 Worker 数后，PS 中心每轮都约为 9.244 GB。gRPC PS 分别约为 4.160 GB 和 2.080 GB/Worker，中心合计都约 8.319 GB。因为数据被更多 Worker 分摊、每个 Worker 同步次数相应减少，**本实验固定总工作量下**的每轮中心字节数近似不变；它并不反驳在“每节点工作量不变、每次同步 M 不变”口径下中心瓶颈随 N 增长的理论。",
        "",
        "## 表二：10 epoch 真实训练的平均单 epoch 指标",
        "",
        "以下每格时间均为十轮的平均，单 epoch 总时间取该轮最慢 Worker；计算/通信时间取 Worker 平均，故两列相加不必等于总时间。加速比统一用 10 epoch 总耗时相除。准确率是**末 epoch 的转向角误差 <0.05 的样本比例**，不是分类准确率，也不是模型已收敛的证明。",
        "",
        "| 配置 | 平均单 epoch 总时间 (s) | 计算 (s) | 通信 (s) | 加速比 | 末轮准确率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in measured:
        lines.append(f"| {row['configuration']} | {row['mean_epoch_total_seconds']:.3f} | {row['mean_epoch_compute_seconds']:.3f} | {row['mean_epoch_communication_seconds']:.3f} | {row['speedup_vs_single']:.3f}× | {row['last_epoch_threshold_accuracy']:.4f} |")
    lines += [
        "",
        "并行效率 = 实测加速比 / Worker 数。原生二进制 Ring 的 2 Worker 效率约 56.5%，4 Worker 约 53.1%；gRPC PS 4 Worker 约 28.5%；Socket PS 4 Worker 约 13.8%。低于理想值的主要原因是梯度序列化与传输、同步等待、CPU 数据准备以及模型更新步数差异；专用 GPU 也不能消除这些开销。",
        "",
        "这是真实单机四卡实验。Socket/gRPC PS 为中心化异步更新，Ring 为同步平均；相同 epoch 的有效 batch 与更新次数不一致，不能把加速比解释为达到相同精度的加速比。",
        "",
        "转向角预测属于回归问题，MAE/RMSE 比阈值准确率更适合主评估。本批历史运行只保存了阈值占比，**无法从占比反推 MAE/RMSE**；依照用户要求不为此重跑，也不对旧数值作虚构修正。若有以前真实采集的 MAE/RMSE，可作为单独历史实验按原配置引用。",
        "",
        "证据：[逐 epoch CSV](../results/multi_gpu_full_v1/aggregate/)、[实测与理想加速比图](../results/multi_gpu_full_v1/aggregate/multi_gpu_speedup.png)、[GPU 使用记录](../results/multi_gpu_full_v1/ring4/gpu_monitor.csv)。",
        "",
    ]
    if args.transport_run_dir:
        lines += ["新增四组 Ring 的[逐轮 CSV 与终端日志](../results/ring_transport_full_v1/)、[十组实测与理想加速比曲线](../results/ring_transport_full_v1/aggregate/all_transport_speedup_curve.png)、[数值校验日志](../results/ring_transport_math_validation.log)。", ""]
    (args.output_dir / "course_two_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "course_two_tables.md")


if __name__ == "__main__":
    main()
