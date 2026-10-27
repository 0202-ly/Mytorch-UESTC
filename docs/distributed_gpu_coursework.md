# 分布式训练、GPU 编程与监控：代码与证据索引

这份索引对应课程任务书。上传的是可运行的实现、复现实验所需脚本，以及从**成功运行**中筛出的轻量 CSV/图；没有上传失败日志、模型权重、大体积原始 Nsight 报告或云服务器凭据。原仓库的 `data/`、`train.txt` 和 `val.txt` 没有被本次提交改写。图像来自仓库现有 DonkeyCar 数据；现有文件数不能证明它是采集时的“完整原始数据集”。

## 任务逐项定位

| 要求 | 代码入口 / 证据 |
|---|---|
| MNIST + MLP 跑通 | [`experiments/mnist_mlp_smoke.py`](../experiments/mnist_mlp_smoke.py)；缺 IDX 文件时可加 `--download`。这是功能 smoke test，不是后面的 DonkeyCar 速度基线。 |
| CNN + DonkeyCar 道路数据 | [`distributed/mytorch_adapter.py`](../distributed/mytorch_adapter.py)、[`model/resnet.py`](../model/resnet.py)、[`distributed/mytorch_single_benchmark.py`](../distributed/mytorch_single_benchmark.py)。 |
| 1→2→4 Worker、Launcher、Monitor、每轮计时 | [`tools/run_multi_gpu_repeat.py`](../tools/run_multi_gpu_repeat.py) 充当 Launcher，启动 PS / Worker；其中 `gpu_monitor` 线程采样 GPU，`tools/aggregate_multi_gpu_repeat.py` 汇总各 Worker 逐轮 CSV。PS、Worker、Launcher、Monitor 四个角色都在真实运行链路中，但 Monitor 不是独立进程。逐轮结果在 [`week4_evidence/ps_epochs/`](week4_evidence/ps_epochs/)，单机基线见 `single_aggregate.csv`。 |
| Socket+JSON PS、2/4 Worker、收敛实验 | [`distributed/socket_ps.py`](../distributed/socket_ps.py)、[`distributed/socket_mytorch_worker.py`](../distributed/socket_mytorch_worker.py)。10 epoch 数据见 `socket2_aggregate.csv`、`socket4_aggregate.csv`。 |
| gRPC+Protobuf PS、2/4 Worker、同接口比较 | [`distributed/training.proto`](../distributed/training.proto)、[`distributed/parameter_server.py`](../distributed/parameter_server.py)、[`distributed/mytorch_grpc_worker.py`](../distributed/mytorch_grpc_worker.py)；`training_pb2*.py` 是同仓库生成的协议代码。10 epoch 数据见 `grpc2_aggregate.csv`、`grpc4_aggregate.csv`。 |
| 每 epoch 计算/通信时间、消息字节数、基线/2/4 Worker、加速比图 | [`week4_evidence/course_measured_epoch_table.csv`](week4_evidence/course_measured_epoch_table.csv)、[`week4_evidence/ps_epochs/multi_gpu_summary.csv`](week4_evidence/ps_epochs/multi_gpu_summary.csv)、[实测与线性加速曲线](week4_evidence/ps_epochs/multi_gpu_speedup.png)；制表与绘图代码在 `tools/aggregate_multi_gpu_repeat.py`、`tools/write_course_tables.py`、`tools/plot_multi_gpu_repeat.py`。通信字节数和计时在同一批逐轮 CSV 中。 |
| Ring AllReduce 的 Split / Reduce-Scatter / All-Gather、2/4 Worker、PS 对比 | [`distributed/mytorch_ring_allreduce.py`](../distributed/mytorch_ring_allreduce.py)、[`tools/verify_ring_math.py`](../tools/verify_ring_math.py)；理论公式、口径提醒和最终对照表见 [`week4_evidence/course_two_tables.md`](week4_evidence/course_two_tables.md)，各传输方式的训练汇总见 [`week4_evidence/ring_transport/`](week4_evidence/ring_transport/)。 |
| 梯度累积、FP16 梯度 | Socket/gRPC Worker 和 Ring 训练入口均有 `--accumulation-steps`；PS 上传梯度与 Ring 网络载荷可用 FP16，参数/更新仍按实现中的 FP32 处理。这是减少同步或字节数的实验设置，不等于已经达到理想线性加速。 |
| NumPy / CuPy / CUDA C 三种卷积 | [`experiments/numpy_cupy_cuda_benchmark.py`](../experiments/numpy_cupy_cuda_benchmark.py)、[`experiments/cuda_conv_benchmark.cu`](../experiments/cuda_conv_benchmark.cu)；[三档输入的执行时间、加速比和误差 CSV](week4_evidence/numpy_cupy_cuda_v2.csv)及[曲线](week4_evidence/numpy_cupy_cuda_time.png)。 |
| GPU 资源监控、step 拆解、Nsight | [`experiments/mytorch_gpu_profile.py`](../experiments/mytorch_gpu_profile.py) 记录 GPU/显存/温度/功耗/CPU、加载、forward、backward、更新与 samples/s；[摘要及原始采样](week4_evidence/gpu_profile/)。Linux 的 [`tools/nsight_profile.sh`](../tools/nsight_profile.sh) 或 Windows 的 [`tools/nsight_profile.ps1`](../tools/nsight_profile.ps1) 可采集独立 CUDA C 卷积的 Nsight Systems 时间线；[已导出的 Kernel/API 汇总](week4_evidence/nsight/)可供核对。 |

## 已保存的关键结果及边界

- 在现有 7860 张训练图、单机四卡环境下，每种配置训练 **10 epoch**。按 10 epoch 总时间相除，单机基线 1.000×，Socket PS 2/4 Worker 为 0.437×/0.552×，gRPC PS 为 0.822×/1.138×，原生二进制 Ring 为 1.130×/2.125×；Socket Ring 为 0.987×/1.453×，gRPC Ring 为 1.284×/2.076×。完整小数和每轮拆分数据见上面的 CSV。速度提高不等于达到了相同预测误差。
- 原结果列名 `accuracy` 是“预测转向角与标签的绝对差小于 **0.05 原始标签单位**的样本比例”，不是分类准确率，也不能从该比例反推 MAE/RMSE。没有把这批数据伪装成 MAE/RMSE；现有数值也不足以证明达到某个指定收敛标准。PS 是中心化更新，Ring 是同步平均，两者有效更新次数与优化轨迹可能不同。
- `course_two_tables.md` 保留了课件的通信量示意，同时明确指出它的 PS/Ring 节点定义与收发方向口径并不完全相同。Ring **单节点**字节数趋近常数，但通信轮次和**全网总字节数**仍随节点数增长。
- 此实现用 CPU/Socket 或 gRPC 传输梯度，没有实现 GPU 直连/NCCL。Nsight Systems 汇总针对**独立 CUDA C 卷积 benchmark**，不是整个 CNN 训练的 Kernel 级归因。CUDA C 与 CuPy 时间为预热后的设备侧时间，未计主机↔设备传输；小卷积下频繁启动和初始化可抵消 Kernel 收益。

## 复现入口

在仓库根目录、具有兼容 CUDA 的 GPU 环境运行。基础依赖见 `requirements-week4.txt`，CuPy 应安装与本机 CUDA 版本对应的发行包，CUDA C 需要 `nvcc`，Nsight 需要 `nsys`。以下命令会写入 Git 忽略的 `results/`，不会覆盖本页证据。

```bash
python -m pip install -r requirements-week4.txt
python -m experiments.mnist_mlp_smoke --download --epochs 1
python -m tools.prepare_distributed_initial_parameters
python -m tools.verify_ring_math
python -m distributed.mytorch_single_benchmark --device cuda --epochs 10 --initial-parameters results/distributed_initial_parameters.npy --output results/single_mytorch.csv
python -m tools.run_multi_gpu_repeat --output-dir results/multi_gpu_repeat --initial-parameters results/distributed_initial_parameters.npy --modes socket2,socket4,grpc2,grpc4,ring2,ring4 --epochs 10 --available-gpus 4
python -m tools.run_multi_gpu_repeat --output-dir results/ring_transport_repeat --initial-parameters results/distributed_initial_parameters.npy --modes ring_socket2,ring_socket4,ring_grpc2,ring_grpc4 --epochs 10 --available-gpus 4
python -m experiments.mytorch_gpu_profile --initial-parameters results/distributed_initial_parameters.npy --output-dir results/mytorch_gpu_profile
```

GPU 卷积（Linux/CUDA）及 Nsight Systems 的入口：

```bash
mkdir -p results
nvcc -O3 experiments/cuda_conv_benchmark.cu -o results/cuda_conv_benchmark
python -m experiments.numpy_cupy_cuda_benchmark --cuda-binary results/cuda_conv_benchmark --output results/numpy_cupy_cuda_benchmark.csv
bash tools/nsight_profile.sh python results/nsight
```

上传的代码和汇总是历史实验的可检查材料；环境、随机数、驱动及通信拓扑变化会影响重跑数值。建议评估回归质量时另存 MAE/RMSE 与划分信息，不改写已有阈值比例数据。
