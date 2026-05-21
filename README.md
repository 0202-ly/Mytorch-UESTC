# MyTorch-UESTC DonkeyCar 自动驾驶实验框架

本仓库是一个面向 DonkeyCar 模拟器图像回归任务的轻量深度学习框架实验项目。项目围绕自研 `MyTorch` 框架展开，完成了张量/自动求导、常用神经网络模块、卷积与池化、数据增强、ResNet18 自动驾驶训练、卷积正确性与性能实验、静态模式融合/图执行器优化、异步数据加载、Docker/Kubernetes/Kubeflow 部署实验，以及答辩 PPT 材料。

当前自动驾驶任务主要预测 **steering angle 转向角**，暂不控制速度。

## 项目结构

```text
.
├── mytorch/                         # 自研深度学习框架核心实现
├── model/                           # MyTorch / PyTorch 模型定义
├── dataset/                         # 示例数据集封装
├── data/                            # DonkeyCar 图像数据，文件名中包含 steering label
├── splits/                          # 时间块划分后的 train/val 列表
├── experiments/                     # 正确性、性能、消融和部署实验脚本
├── deploy/                          # Docker / Kubernetes 部署文件
├── kuflow/                          # Kubeflow/Kuflow 风格算子与 pipeline 描述
├── results/                         # 轻量实验结果与答辩 PPT
├── tools/                           # 本地辅助工具
├── train_donkeycar_resnet18_jit_vs_pytorch.py
├── benchmark_donkeycar_resnet18_jit.py
├── benchmark_jit_training_batch_breakdown.py
├── auto_drive.py
├── test_*.py
├── train.txt / val.txt
└── README.md
```

## 根目录文件说明

| 文件 | 作用 |
|---|---|
| `auto_drive.py` | 自动驾驶推理/运行入口，用训练好的模型对图像输入预测转向角。 |
| `train_donkeycar_resnet18_jit_vs_pytorch.py` | DonkeyCar ResNet18 主训练脚本，支持 MyTorch、MyTorch JIT 风格图执行、PyTorch 对比，并支持自定义 `train-list` / `val-list`。 |
| `benchmark_donkeycar_resnet18_jit.py` | MyTorch 内部融合消融主脚本，对比无融合、静态模式融合、静态融合+图执行器等训练性能与 Val MSE。 |
| `benchmark_jit_training_batch_breakdown.py` | 训练 batch 级耗时拆解脚本，用于分析 forward、backward、loader 等阶段耗时。 |
| `test_all_function.py` | MyTorch 基础功能综合测试入口。 |
| `test_grad.py` | 自动求导与梯度相关测试。 |
| `test_jit_training.py` | JIT 风格训练路径 smoke test。 |
| `test_jit_training_correctness.py` | JIT/融合训练路径的正确性验证。 |
| `train.txt` / `val.txt` | 当前默认训练集、验证集列表。 |
| `train.txt.random_frame_backup` / `val.txt.random_frame_backup` | 原随机帧划分备份，用于对比随机划分与时间块划分差异。 |
| `.gitignore` | 排除大模型权重、缓存、完整训练输出等不适合上传 GitHub 的文件。 |
| `.dockerignore` | Docker 构建时排除无关文件。 |

## `mytorch/` 框架核心

| 文件 | 作用 |
|---|---|
| `tensor.py` | MyTorch 张量对象与基础属性封装。 |
| `function.py` | 自动求导 Function 与核心算子实现，包括卷积、池化、矩阵运算、激活等。 |
| `modules.py` | 神经网络模块层封装，如 Linear、Conv2d、BatchNorm、ReLU、Pooling、Sequential 等。 |
| `loss.py` | 损失函数实现，用于分类/回归训练。 |
| `optim.py` | 优化器实现，如 SGD/Adam 等。 |
| `dataset.py` | DonkeyCar 数据集读取、`train-list` / `val-list` 支持、图像与转向角标签加载。 |
| `dataloader.py` | 同步 DataLoader 实现。 |
| `async_dataloader.py` | 生产者-消费者式异步数据加载实现，支持 workers 和 prefetch。 |
| `transforms.py` | 数据处理与数据增强模块，包括翻转、亮度/对比度/gamma、噪声、HSV、中心裁剪、MixUp/CutMix 等实验接口。 |
| `jit.py` | JIT 风格图执行入口。注意：本项目未实现完整通用 JIT 编译器，这里主要是静态融合后的图执行调度。 |
| `jit_ir.py` | 计算图/IR 相关结构实验代码。 |
| `jit_pass.py` | 图优化 pass 与模式处理相关实验代码。 |
| `jit_codegen.py` | 代码生成方向的实验文件，当前不是完整运行时代码生成器。 |
| `jit_train_rewrite.py` | 训练期静态模式融合重写逻辑，如 BN+ReLU、BN+Add+ReLU、Add+ReLU。 |
| `constant_folding.py` | 常量折叠等计算图优化实验。 |
| `mixed_precision.py` | 混合精度相关实验实现。 |
| `dataset_analyzer.py` | 数据集统计与分析工具。 |
| `utils.py` | 通用工具函数。 |
| `test_DataModules.py` / `test_gradient.py` | MyTorch 局部测试脚本。 |

## `model/` 模型文件

| 文件 | 作用 |
|---|---|
| `resnet.py` | MyTorch ResNet/ResNet18 风格模型定义，用于 DonkeyCar 图像回归实验。 |
| `autodrive_net.py` | MyTorch 自动驾驶网络定义。 |
| `autodrive_net_pytorch.py` | PyTorch 自动驾驶网络定义，用于框架对照实验。 |
| `lenet.py` | LeNet 示例模型。 |
| `iris_model.py` | Iris 示例模型。 |

## 数据与划分

| 路径 | 作用 |
|---|---|
| `data/` | DonkeyCar 图像数据集。图片文件名格式类似 `frame_steering.jpg`，其中 steering 值作为回归标签来源。 |
| `dataset/` | 示例数据集类，包括 `autodrive_dataset.py`、`iris_dataset.py`、`mnist_dataset.py`。 |
| `splits/temporal_block_gap20/train.txt` | 时间块划分后的训练集列表。 |
| `splits/temporal_block_gap20/val.txt` | 时间块划分后的验证集列表。 |
| `splits/temporal_block_gap20/split_report.md` | 时间块划分报告，记录验证集附近 purge gap 后的近邻泄漏情况。 |

时间块划分用于减少 DonkeyCar 连续视频帧随机划分带来的相邻帧泄漏。重新生成划分可运行：

```powershell
python experiments\make_temporal_split.py --data-root . --output-dir splits\temporal_block_gap20 --purge-gap 20
```

## `experiments/` 实验脚本

| 文件 | 作用 |
|---|---|
| `conv_pool_correctness.py` | 卷积/池化正确性实验，对比 NumPy naive、PyTorch、MyTorch，在 stride/padding/dilation/groups 组合下验证 forward/backward/gradcheck 误差。 |
| `conv_performance.py` | 卷积性能实验，对比 naive conv、im2col+GEMM、PyTorch Conv2d，统计 latency、FLOPS、显存等。 |
| `conv_arch_ablation.py` | 卷积结构消融实验的 PyTorch 版本/早期版本。 |
| `conv_arch_ablation_mytorch.py` | MyTorch 卷积结构消融实验，包括 kernel size、depthwise、groups、dilation、transpose conv 等变体。 |
| `augmentation_ablation.py` | 数据增强消融实验，支持 none、单项增强、MixUp、CutMix、DonkeyCar 安全增强等，并输出验证集曲线和增强样例图。 |
| `loader_ablation.py` | 同步/异步 DataLoader 消融实验，对比 loader fetch、epoch time、吞吐等。 |
| `benchmark_donkeycar_deployment_modes.py` | DonkeyCar 本地、Docker、Kubernetes Job 等部署模式 benchmark。 |
| `make_temporal_split.py` | 生成 temporal block split，并支持 purge gap 去除验证集相邻训练帧。 |

常用实验命令示例：

```powershell
# 数据增强消融
python experiments\augmentation_ablation.py --data-root . --train-list splits\temporal_block_gap20\train.txt --val-list splits\temporal_block_gap20\val.txt --epochs 20 --batch-size 64

# MyTorch 卷积结构消融
python experiments\conv_arch_ablation_mytorch.py --data-root . --train-list splits\temporal_block_gap20\train.txt --val-list splits\temporal_block_gap20\val.txt --epochs 20

# JIT/静态融合训练消融
python benchmark_donkeycar_resnet18_jit.py --data-root . --train-list splits\temporal_block_gap20\train.txt --val-list splits\temporal_block_gap20\val.txt --results-dir results\jit_fusion_ablation_full_bs32 --epochs 5 --batch-size 32
```

## JIT 与静态融合说明

本项目中的 `JIT Fused` 不是完整通用 JIT 编译器。当前没有实现通用动态图捕获、完整 IR lowering、运行时代码生成和新 kernel 编译。

实际完成的是一个轻量替代路径：

1. 在固定 DonkeyCar ResNet18 训练图上扫描已知模式。
2. 对语义清晰的局部模式做静态融合，如 `BN+ReLU`、`BN+Add+ReLU`、`Add+ReLU`。
3. 使用图执行器按照融合后的执行计划运行。
4. 用 MyTorch 内部基线进行消融：无融合、静态模式融合、静态融合+图执行器。

对应结果位于：

```text
results/jit_fusion_ablation_full_bs32/
├── training_fusion_ablation.csv
├── training_fusion_ablation.md
├── epoch_records.csv
└── benchmark_summary.json
```

## `deploy/` 部署文件

| 文件 | 作用 |
|---|---|
| `deploy/docker/Dockerfile` | 原始/通用 Dockerfile。 |
| `deploy/docker/Dockerfile.donkeycar` | DonkeyCar 训练任务专用 Dockerfile。 |
| `deploy/docker/docker-compose.yml` | 原始 docker-compose 配置。 |
| `deploy/docker/docker-compose.donkeycar.yml` | DonkeyCar 训练任务 docker-compose 配置。 |
| `deploy/docker/requirements-donkeycar.txt` | Docker 环境内 DonkeyCar 训练所需依赖。 |
| `deploy/k8s/donkeycar-training-job.yaml` | Kubernetes DonkeyCar 训练 Job。 |
| `deploy/k8s/donkeycar-training-job.kind.yaml` | kind 本地集群适配版训练 Job。 |
| `deploy/k8s/donkeycar-pv-pvc.kind.yaml` | kind 环境下的数据挂载 PV/PVC 配置。 |
| `deploy/k8s/kind-donkeycar.yaml` | kind 集群配置。 |
| `deploy/k8s/mnist-training-job.yaml` | MNIST 示例 Job。 |
| `deploy/k8s/README.md` | Kubernetes 部署说明。 |

## `kuflow/` Kubeflow/Kuflow 风格文件

| 文件 | 作用 |
|---|---|
| `operators.py` | 训练/推理算子封装，描述 DonkeyCar 训练任务如何以 operator 形式启动。 |
| `kfp_pipeline.py` | Kubeflow Pipeline 描述与 YAML 生成入口，支持在没有完整 KFP 环境时输出任务规格说明。 |
| `__init__.py` | Python 包标记文件。 |

## `results/` 结果与 PPT

仓库只保留了轻量结果和答辩材料。完整训练输出、模型权重、缓存文件已由 `.gitignore` 排除。

| 路径 | 作用 |
|---|---|
| `results/defense_ppt/` | 答辩 PPT、历史版本和部分预览图。 |
| `results/jit_fusion_ablation_full_bs32/` | JIT/静态融合消融的关键 CSV/Markdown/JSON 结果。 |

较新的答辩版本包括：

```text
results/defense_ppt/DonkeyCar自动驾驶框架答辩-v20.pptx
results/defense_ppt/DonkeyCar自动驾驶框架答辩-v14.pptx
```

## 运行环境建议

建议使用已有的 `donkey-env` Conda 环境：

```powershell
conda activate donkey-env
```

常用依赖包括：

- Python
- NumPy
- OpenCV
- PyTorch
- Matplotlib
- pandas
- scikit-learn
- Docker / kubectl / kind，部署实验需要

## GitHub 上传说明

由于训练结果和模型权重较大，`.gitignore` 默认排除了：

- `results/` 中的大型训练输出
- `*.pt` / `*.pth` / `*.pkl` / `*.onnx` / `*.npz`
- Python 缓存和 CuPy 缓存
- 虚拟环境目录

如果需要保存模型权重，建议使用 Git LFS 或单独上传到网盘/Release，而不是直接提交到普通 GitHub 仓库。
