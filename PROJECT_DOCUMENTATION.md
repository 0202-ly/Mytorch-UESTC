# 项目整理与代码说明文档

本文档用于说明本项目的整体框架、主要执行流程、每个源码文件的功能和实现方式。项目主体是一个围绕 DonkeyCar 自动驾驶图像回归任务构建的自研轻量深度学习框架实验工程。

## 1. 项目定位

项目名称可理解为 **MyTorch-UESTC DonkeyCar 自动驾驶实验框架**。

核心目标：

- 实现一个自研深度学习框架 `mytorch`，包含 Tensor、自动求导、神经网络层、损失函数、优化器、数据加载和部分 JIT/图优化能力。
- 使用 DonkeyCar 图像数据进行转向角回归训练，主要预测 `steering angle`，当前不训练油门控制。
- 对比 MyTorch 与 PyTorch 的训练效果、速度、资源占用。
- 做卷积/池化正确性、卷积性能、数据增强、异步 DataLoader、静态融合/JIT、Docker/Kubernetes/Kubeflow 部署等实验。

整体数据流：

```text
data/*.jpg + train.txt/val.txt
        |
        v
Dataset / DataLoader / AsyncDataLoader
        |
        v
MyTorch Tensor + Transform
        |
        v
ResNet18 / AutoDriveNet
        |
        v
MSELoss
        |
        v
Tensor.backward() 自动求导
        |
        v
Optimizer.step()
        |
        v
results/*.csv / *.json / *.md / model state
```

训练和部署实验的工程流：

```text
本地脚本
  -> train_donkeycar_resnet18_jit_vs_pytorch.py
  -> benchmark_donkeycar_resnet18_jit.py

容器化
  -> deploy/docker/Dockerfile.donkeycar
  -> deploy/docker/docker-compose.donkeycar.yml

集群运行
  -> deploy/k8s/*.yaml
  -> kuflow/operators.py
  -> kuflow/kfp_pipeline.py
```

## 2. 目录结构

```text
.
├── mytorch/                         # 自研深度学习框架核心
├── model/                           # MyTorch / PyTorch 模型定义
├── dataset/                         # 示例数据集封装
├── data/                            # DonkeyCar 图像数据
├── splits/                          # 时间块 train/val 划分
├── experiments/                     # 正确性、性能、消融、部署实验
├── deploy/                          # Docker / Kubernetes 部署配置
├── kuflow/                          # Kubeflow/Kuflow 风格算子与 pipeline
├── tools/                           # 本地辅助工具
├── auto_drive.py                    # DonkeyCar 推理/仿真运行脚本
├── train_donkeycar_resnet18_jit_vs_pytorch.py
├── benchmark_donkeycar_resnet18_jit.py
├── benchmark_jit_training_batch_breakdown.py
├── test_*.py
├── train.txt / val.txt
└── README.md
```

## 3. 核心框架设计

### 3.1 Tensor 与自动求导

`mytorch/tensor.py` 定义 `Tensor`。它封装 NumPy 或 CuPy 数组，并记录：

- `data`：真实数据。
- `grad`：梯度。
- `creator`：生成该 Tensor 的 `Function` 节点。
- `requires_grad`：是否参与求导。
- `_device`：`cpu` 或 `cuda`。

实现方式：

- 初始化时根据输入数组类型和 `device` 参数选择 NumPy/CuPy 后端。
- `cuda()` / `cpu()` 会迁移 `data` 和 `grad`。
- `xp` 属性返回当前后端库，算子可统一调用 `np` 或 `cp`。
- `backward()` 使用拓扑排序从损失 Tensor 反向遍历计算图，逐个调用 `Function.backward()`。
- `add_()`、`mul_()`、`addcmul_()`、`addcdiv_()` 等原地操作会检查非叶子节点，避免破坏计算图。

### 3.2 Function 算子系统

`mytorch/function.py` 定义自动求导算子基类 `Function` 和大量具体算子。

实现方式：

- `Function.__call__()` 先检查所有输入 Tensor 是否在同一设备，再执行 `forward()`。
- 前向结果是 Tensor 时，算子会作为 `creator` 参与自动求导图。
- JIT tracing 开启时，`Function.__call__()` 同时创建 `IRNode`，记录算子名、输入、输出 shape、dtype 和原始算子对象。
- 每个算子类实现自己的 `forward()`、`backward()` 和 `_get_inputs()`。

重点算子：

| 算子 | 功能 | 实现方式 |
|---|---|---|
| `Add` | 张量加法 | 支持广播梯度回传 |
| `MatMul` | 矩阵乘法 | 前向调用矩阵乘法，反向按矩阵求导公式计算 |
| `Conv2dOp` | 2D 卷积 | 使用 `im2col + GEMM`，支持 stride、padding、dilation、groups |
| `ConvTranspose2dOp` | 转置卷积 | 使用扩展 `im2col/col2im` 逻辑处理反卷积形状 |
| `MaxPoolOp` / `AvgPoolOp` / `MinPoolOp` | 池化 | 保存池化位置或窗口统计用于反向传播 |
| `BatchNorm2dOp` | 2D BatchNorm | 训练态更新 running mean/var，评估态使用缓存统计 |
| `MSE` | 均方误差 | 用于 DonkeyCar 转向角回归 |
| `LogSoftmaxOp` / `NLLLossOp` | 分类损失基础算子 | 用于 CrossEntropy |
| `Fused*Op` | 融合算子 | 将常见子图合并，减少 Python 调度和中间 Tensor |

### 3.3 Module 层系统

`mytorch/modules.py` 定义类似 PyTorch 的 `Module` 体系。

实现方式：

- `Module.parameters()` 递归收集子模块、列表、元组、字典中的可训练 Tensor。
- `cuda()` / `cpu()` 递归迁移所有参数。
- `train()` / `eval()` 递归切换模块状态。
- `save_weights()` / `load_weights()` 使用 pickle 保存和恢复参数列表。

主要层：

| 层 | 功能 | 实现方式 |
|---|---|---|
| `Linear` | 全连接层 | Xavier 初始化，调用 `MatMul + Add` |
| `Conv2d` | 标准/分组/空洞卷积 | He 风格初始化，调用 `Conv2dOp` |
| `DepthwiseSeparableConv2d` | 深度可分离卷积 | depthwise group conv + pointwise 1x1 conv |
| `ConvTranspose2d` | 转置卷积 | 调用 `ConvTranspose2dOp` |
| `Flatten` | 拉平 | 调用 `ReshapeOp` |
| `ReLU` / `ELU` / `Sigmoid` | 激活函数 | 调用对应 Function |
| `MaxPool` / `AvgPool` / `MinPool` | 池化层 | 调用对应 Pool Op |
| `AdaptiveAvgPool2d` | 自适应平均池化 | 根据输入输出大小推导 kernel/stride |
| `BatchNorm2d` | 批归一化 | 使用 `BatchNorm2dOp`，维护 running 统计 |
| `FusedBatchNormReLU` | BN + ReLU 融合 | 复用原 BN 参数和 running 统计 |
| `FusedBatchNormAddReLU` | BN + 残差 Add + ReLU | 用于 ResNet BasicBlock 残差尾部 |
| `FusedConvBNReLU` / `FusedConvBNAddReLU` | Conv/BN/Residual/ReLU 融合 | 用于图优化和实验性融合路径 |

### 3.4 数据系统

项目有两套数据集位置：

- `mytorch/dataset.py`：框架内置数据集基类、MNIST、AutoDriveDataset。
- `dataset/*.py`：示例或任务级数据集封装。

DonkeyCar 数据格式：

```text
./data/1000_-0.0222.jpg -0.0222
./data/1001_-0.0222.jpg -0.0222
```

第一列是图片相对路径，第二列是 steering label。图片文件名本身也包含帧号和转向角，`experiments/make_temporal_split.py` 会使用这个约定生成时间块划分。

### 3.5 JIT 与静态融合

本项目中的 JIT 不是完整通用编译器。当前更准确地说，是“trace + IR + pattern pass + fused graph executor / CUDA RawKernel 实验”的组合。

实现分层：

| 文件 | 职责 |
|---|---|
| `mytorch/jit_ir.py` | 定义 `IRNode`、`IRGraph`、`TracerState` |
| `mytorch/jit.py` | `CompiledModule`，负责 trace、cache、训练/推理执行路径 |
| `mytorch/jit_pass.py` | 常量折叠、无效 Add/Reshape/ReLU 删除、BN/ReLU/Conv/Add 模式融合 |
| `mytorch/jit_codegen.py` | 实验性 CUDA kernel 代码生成 |
| `mytorch/jit_train_rewrite.py` | 对 ResNet 训练结构做静态替换，把 BasicBlock 改为融合块 |

训练融合路径：

1. 检测 ResNet stem 中的 `bn1 + relu`。
2. 检测 BasicBlock 中的 `conv1 -> bn1 -> relu1` 和 `conv2 -> bn2 -> add -> relu2`。
3. 替换为 `FusedBatchNormReLU` 和 `FusedBatchNormAddReLU`。
4. 复用原 BatchNorm 参数，不改变优化器可见的 Tensor。

图优化路径：

1. 执行 eager forward 得到真实输出。
2. tracing 时记录 IR 节点。
3. `optimize_graph()` 做保守模式融合。
4. 缓存相同输入 shape/device/dtype 的执行计划。

## 4. 根目录文件说明

| 文件 | 功能 | 实现方式 |
|---|---|---|
| `README.md` | 项目已有说明 | 概述项目结构、实验命令、部署说明和 GitHub 上传注意事项 |
| `__init__.py` | 根包标记 | 空文件，用于把根目录作为 Python package |
| `auto_drive.py` | DonkeyCar 仿真推理入口 | 创建 `gym_donkeycar` 环境，加载模型权重，循环读取 frame，预处理为 NCHW Tensor，模型预测 steering，固定 throttle 执行动作 |
| `train_donkeycar_resnet18_jit_vs_pytorch.py` | DonkeyCar 主训练和对比脚本 | 解析实验配置，构建 MyTorch/PyTorch 数据集和模型，执行训练/验证，采集 batch/epoch 指标和资源占用，输出 JSON/CSV/Markdown |
| `benchmark_donkeycar_resnet18_jit.py` | JIT/静态融合消融 benchmark | 对比原始 MyTorch、训练融合、JIT 图执行、实验性 Conv-BN 融合等模式，统计训练速度、验证误差、显存和加速比 |
| `benchmark_jit_training_batch_breakdown.py` | 单 batch 训练耗时拆解 | 构造固定 batch 和模型变体，分别计时 forward、loss、backward、optimizer 等阶段 |
| `test_all_function.py` | 基础算子梯度综合测试 | 用数值梯度检查 MyTorch 多个 Function 的反向传播，并可生成答辩图 |
| `test_grad.py` | 梯度检查示例 | 构造小网络，通过 `mytorch.utils.grad_check_model` 验证梯度精度 |
| `test_jit_training.py` | JIT 训练 smoke test | 用 TinyMLP、TinyConvBN 验证 JIT training forward graph 是否保留 autograd creator 和融合行为 |
| `test_jit_training_correctness.py` | JIT 训练正确性测试 | 对 eager 与 JIT 单步训练的输出、梯度、参数更新进行对比，覆盖 MLP、ConvBN、Residual、ResNet BasicBlock |
| `train.txt` | 默认训练列表 | 每行图片路径和 steering label |
| `val.txt` | 默认验证列表 | 每行图片路径和 steering label |
| `train.txt.random_frame_backup` | 随机划分训练列表备份 | 时间块划分启用前的备份 |
| `val.txt.random_frame_backup` | 随机划分验证列表备份 | 时间块划分启用前的备份 |
| `.gitignore` | Git 忽略规则 | 排除缓存、大型训练输出、模型权重、虚拟环境等 |
| `.dockerignore` | Docker 构建忽略规则 | 减少 Docker build context 中的无关文件 |

使用注意：`auto_drive.py` 当前使用 `model = ResNet()`，但 `model/resnet.py` 中 `ResNet` 类需要传入 block 和 num_blocks；当前兼容入口是 `ResNet18(...)`、`ResNet18Original(...)`、`ResNet18Fused(...)`。如果要运行 `auto_drive.py`，应先把模型构造改成当前接口，并确认权重格式匹配。

## 5. `mytorch/` 代码说明

| 文件 | 功能 | 实现方式 |
|---|---|---|
| `mytorch/__init__.py` | 框架统一导出入口 | 从 `tensor`、`modules`、`loss`、`optim`、`dataloader`、`dataset` 导出常用类，方便 `from mytorch import Tensor, Module` |
| `mytorch/tensor.py` | Tensor 和自动求导入口 | 封装 NumPy/CuPy 数组、设备迁移、梯度、creator、拓扑反向传播和安全原地操作 |
| `mytorch/function.py` | 自动求导算子和融合算子 | 每个算子继承 `Function`，实现前向、反向、输入回溯；卷积采用 im2col/GEMM；融合算子减少中间节点和调度开销 |
| `mytorch/modules.py` | 神经网络层和 Module 基类 | 递归收集参数、切换训练/评估、迁移设备；封装 Linear、Conv、Pool、BN、激活和融合层 |
| `mytorch/loss.py` | 损失函数 | `MSELoss` 调用普通或融合 MSE；`CrossEntropyLoss` 可调用融合 CE 或 LogSoftmax+NLL |
| `mytorch/optim.py` | 优化器和学习率调度器 | 实现 `Optimizer` 参数组、状态字典、SGD、Momentum、Adagrad、Rmsprop、Adam，以及多种 LR scheduler |
| `mytorch/dataloader.py` | 同步 DataLoader | 按 batch 切分索引，支持 shuffle 和默认 collate，将样本堆叠为 Tensor |
| `mytorch/async_dataloader.py` | 异步 DataLoader | 使用线程、队列和 prefetch，在后台生产 batch；`ParallelDataLoader` 当前用异步线程版作为基础 |
| `mytorch/dataset.py` | 框架内置数据集 | 定义 `Dataset` 基类、MNIST 下载/校验/解析、DonkeyCar AutoDriveDataset |
| `mytorch/transforms.py` | 数据增强和变换 | 实现 Compose、Normalize、Crop、Flip、Rotation，以及 DonkeyCar 标签感知增强、MixUp、CutMix、LocalMixUp |
| `mytorch/dataset_analyzer.py` | 数据集分析工具 | 统计类别分布、特征均值方差，生成增强和归一化建议报告 |
| `mytorch/utils.py` | 通用工具 | 数值梯度检查、计算图可视化 DOT 输出、紧凑图输出 |
| `mytorch/jit_ir.py` | JIT IR 数据结构 | `IRNode` 表示算子节点，`IRGraph` 保存节点和输出，`TracerState` 管理 tracing 状态 |
| `mytorch/jit.py` | JIT 编译包装器 | `CompiledModule` 包装模型，根据训练/推理状态 trace 图、优化图、缓存快路径、委托参数和模式切换 |
| `mytorch/jit_pass.py` | 图优化 pass | 统计消费者，做常量折叠、零加消除、重复 ReLU 消除、Conv/BN/ReLU/Add 融合 |
| `mytorch/jit_codegen.py` | CUDA 代码生成实验 | 根据 IR 节点生成部分 CUDA kernel 字符串，如 Add、ReLU、AddReLU、BNReLU、Conv2d |
| `mytorch/jit_train_rewrite.py` | 训练期静态融合重写 | 按结构识别 ResNet BasicBlock，替换为 `DynamicFusedBasicBlock`，并融合 stem |
| `mytorch/constant_folding.py` | 推理期 Conv-BN 折叠 | 将 Conv + BN 的权重和偏置折叠到 Conv 中，支持直接属性和 list 容器 |
| `mytorch/mixed_precision.py` | 混合精度推理实验 | 将 Tensor 和模块参数转为 FP16，可选择保留首层 Conv 和末层 Linear 为 FP32，并可先折叠 BN |
| `mytorch/test_DataModules.py` | DataLoader 局部测试 | 构造 TupleDataset，测试 shuffle 和 batch collate |
| `mytorch/test_gradient.py` | 单元测试式梯度检查 | 用 unittest 检查 ReLU、Conv2d、MaxPool 的数值梯度 |

## 6. `model/` 代码说明

| 文件 | 功能 | 实现方式 |
|---|---|---|
| `model/resnet.py` | MyTorch ResNet18 | 定义 `Downsample`、原始 `BasicBlockOriginal`、融合 `BasicBlockFullFusion`、通用 `ResNet` 组装类，以及 `ResNet18Original/Fused/ResNet18` 工厂函数 |
| `model/autodrive_net.py` | MyTorch 自动驾驶 CNN | Nvidia PilotNet 风格结构，5 个卷积层 + ELU + Flatten + 多层 Linear，输出 1 维转向角 |
| `model/autodrive_net_pytorch.py` | PyTorch 自动驾驶 CNN | 与 `AutoDriveNet` 对齐的 PyTorch `nn.Sequential` 实现，用于框架对比实验 |
| `model/lenet.py` | MyTorch LeNet 示例 | 使用 Conv/Pool/Flatten/Linear 实现 MNIST 风格分类模型 |
| `model/iris_model.py` | Iris 示例模型 | 提供简单神经网络和分类器封装，包含 softmax、predict、evaluate 等辅助逻辑 |

`model/resnet.py` 的核心结构：

```text
Conv7x7 -> BN/ReLU 或 FusedBNReLU -> MaxPool
  -> layer1: 2 个 BasicBlock, 64 通道
  -> layer2: 2 个 BasicBlock, 128 通道, stride=2
  -> layer3: 2 个 BasicBlock, 256 通道, stride=2
  -> layer4: 2 个 BasicBlock, 512 通道, stride=2
  -> AdaptiveAvgPool2d(1,1)
  -> Flatten
  -> Linear(512, num_classes)
```

DonkeyCar 任务中 `num_classes` 实际作为 `output_dim` 使用，通常为 1。

## 7. `dataset/` 代码说明

| 文件 | 功能 | 实现方式 |
|---|---|---|
| `dataset/autodrive_dataset.py` | 任务级 DonkeyCar 数据集 | 读取 train/val list，OpenCV 加载图片，BGR 转 RGB，支持标签感知 transform，输出 `(image_tensor, label_tensor)` |
| `dataset/iris_dataset.py` | Iris 数据集 | 使用 sklearn/pandas 读取和划分 Iris，做标准化，输出 MyTorch Tensor |
| `dataset/mnist_dataset.py` | MNIST 数据集 | 下载/校验 gzip idx 文件，解析图像和标签，输出训练或测试样本 |

说明：`dataset/autodrive_dataset.py` 与 `mytorch/dataset.py` 中的 `AutoDriveDataset` 功能接近。前者更像任务目录下的增强版本，支持 transform 同时处理图片和标签；后者是框架内置版本。

## 8. `experiments/` 代码说明

| 文件 | 功能 | 实现方式 |
|---|---|---|
| `experiments/make_temporal_split.py` | 生成时间块 train/val 划分 | 从已有列表或 `data/*.jpg` 解析 frame id，按 block 划分验证集，并 purge 验证集附近训练帧，降低相邻帧泄漏 |
| `experiments/augmentation_ablation.py` | 数据增强消融 | 使用 PyTorch 训练 AutoDriveNet，比较 none、单项增强、MixUp、CutMix、DonkeyCar 安全增强，输出曲线、样例图、CSV/Markdown |
| `experiments/conv_arch_ablation.py` | PyTorch 卷积结构消融 | 定义可配置 ResNet18 变体，比较 kernel size、depthwise、groups、dilation、transpose conv 等 |
| `experiments/conv_arch_ablation_mytorch.py` | MyTorch 卷积结构消融 | 用 MyTorch 实现同类 ResNet 变体，统计参数量、FLOPs、训练/验证指标和 forward latency |
| `experiments/conv_performance.py` | 卷积性能 benchmark | 构造不同卷积 case，对比 naive conv、MyTorch im2col/GEMM、PyTorch Conv2d，输出 latency/FLOPs/内存 |
| `experiments/conv_pool_correctness.py` | 卷积/池化正确性验证 | 用 NumPy naive、PyTorch、MyTorch 三方对比 forward/backward 和数值梯度误差 |
| `experiments/loader_ablation.py` | DataLoader 消融 | 对比同步和异步 MyTorch loader，监控 GPU 利用率、epoch 时间、fetch 时间和吞吐 |
| `experiments/benchmark_donkeycar_deployment_modes.py` | 部署模式 benchmark | 调用本地命令、Docker、Kubernetes job，记录 wall time、返回码、输出摘要并生成结果图 |

## 9. `deploy/` 配置说明

| 文件 | 功能 | 实现方式 |
|---|---|---|
| `deploy/docker/Dockerfile` | 通用 Docker 镜像 | 基于 Python 镜像安装依赖并复制项目 |
| `deploy/docker/Dockerfile.donkeycar` | DonkeyCar 训练镜像 | 基于 `python:3.11-slim`，安装 `requirements-donkeycar.txt`，复制项目，默认执行 MyTorch JIT CPU 训练命令 |
| `deploy/docker/requirements-donkeycar.txt` | 容器依赖 | 列出 DonkeyCar 训练所需 Python 包 |
| `deploy/docker/docker-compose.yml` | 通用 compose | 提供基础容器运行配置 |
| `deploy/docker/docker-compose.donkeycar.yml` | DonkeyCar compose | 构建 `mytorch-donkeycar:latest`，挂载数据和输出目录，通过环境变量配置 epochs、batch size、loader 等 |
| `deploy/docker/.dockerignore` | deploy/docker 局部忽略 | 减少 Docker 构建上下文 |
| `deploy/k8s/README.md` | Kubernetes 部署说明 | 说明镜像构建、PVC 准备和 Job 提交流程 |
| `deploy/k8s/donkeycar-training-job.yaml` | Kubernetes DonkeyCar Job | 使用 PVC 挂载 `/data/donkeycar` 和 `/outputs`，运行训练脚本 |
| `deploy/k8s/donkeycar-training-job.kind.yaml` | kind 本地 Job | 使用 `hostPath` 指向 `/workspace`，适合本地 kind 集群验证 |
| `deploy/k8s/donkeycar-pv-pvc.kind.yaml` | kind PV/PVC | 给本地 kind 提供数据和输出卷 |
| `deploy/k8s/kind-donkeycar.yaml` | kind 集群配置 | 定义本地 kind cluster |
| `deploy/k8s/mnist-training-job.yaml` | MNIST 示例 Job | 保留为 Kubernetes smoke test |

## 10. `kuflow/` 代码说明

| 文件 | 功能 | 实现方式 |
|---|---|---|
| `kuflow/__init__.py` | 包导出 | 导出 operator 相关对象 |
| `kuflow/operators.py` | 训练任务算子封装 | 定义 `TaskOperator`、资源规格、启动规格，生成 DonkeyCar/MNIST 训练 argv 和 Kubernetes Job dict |
| `kuflow/kfp_pipeline.py` | Kubeflow Pipeline 生成 | 如果安装了 `kfp`，编译 pipeline；否则生成手写 fallback Argo/Kubeflow YAML，保证无 KFP SDK 时也能产出部署描述 |

## 11. 数据、划分、工具和结果目录

| 路径 | 功能 | 说明 |
|---|---|---|
| `data/` | DonkeyCar 图片数据 | 文件名形如 `1000_-0.0222.jpg`，包含帧号和 steering label |
| `splits/temporal_block_gap20/train.txt` | 时间块训练列表 | 由 `make_temporal_split.py` 生成 |
| `splits/temporal_block_gap20/val.txt` | 时间块验证列表 | 由 `make_temporal_split.py` 生成 |
| `splits/temporal_block_gap20/split_report.md` | 划分报告 | 记录划分参数、样本数、purge 后近邻统计 |
| `tools/kind.exe` | 本地 kind 工具 | 用于 Windows 环境本地 Kubernetes 集群实验 |
| `results/` | 实验输出和答辩材料 | `.gitignore` 默认只保留轻量结果和 PPT，过滤大型权重/训练输出 |

## 12. 主要运行流程

### 12.1 本地 MyTorch/PyTorch 对比训练

入口：`train_donkeycar_resnet18_jit_vs_pytorch.py`

流程：

1. `parse_args()` 读取数据目录、列表、backend、batch size、epochs、设备、loader 和 JIT 参数。
2. `ExperimentConfig` 保存完整配置。
3. `make_mytorch_loaders()` 或 `make_torch_loaders()` 构建数据加载器。
4. MyTorch 路径构建 `ResNet18Original(output_dim)`，可通过 `mytorch.jit` 包装。
5. PyTorch 路径构建等价 ResNet18。
6. 每个 epoch 执行训练、验证、资源采样。
7. `finalize_backend_result()` 汇总训练集/验证集指标、速度、FLOPs 和资源信息。
8. 输出 JSON、batch CSV、epoch CSV、对比 Markdown。

### 12.2 JIT/融合消融

入口：`benchmark_donkeycar_resnet18_jit.py`

流程：

1. 构建统一配置和数据加载器。
2. 分别运行不同训练变体。
3. 对比无融合、静态训练融合、JIT 图执行、实验性 Conv-BN 融合等。
4. 对每个变体记录 epoch 时间、batch 时间、loss、Val MSE、资源占用。
5. 生成 summary JSON、CSV 和 Markdown，并计算 speedup。

### 12.3 卷积/池化正确性验证

入口：`experiments/conv_pool_correctness.py`

流程：

1. 构造 stride、padding、dilation、groups 等 case。
2. 分别用 NumPy naive、PyTorch、MyTorch 计算 forward/backward。
3. 用数值梯度验证 MyTorch backward。
4. 输出 CSV 和 Markdown 报告。

### 12.4 时间块数据划分

入口：`experiments/make_temporal_split.py`

流程：

1. 从 `train.txt`/`val.txt` 或 `data/*.jpg` 读取所有样本。
2. 从文件名或列表中解析 `frame_id` 和 label。
3. 按 block 选择验证集。
4. 删除验证帧附近 `purge_gap` 范围内的训练帧。
5. 写入新的 train/val list 和 `split_report.md`。
6. 可选 `--activate` 将新列表复制到根目录并备份旧随机划分。

## 13. 常用命令

生成时间块划分：

```powershell
python experiments\make_temporal_split.py --data-root . --output-dir splits\temporal_block_gap20 --purge-gap 20
```

运行 MyTorch/PyTorch 对比训练：

```powershell
python train_donkeycar_resnet18_jit_vs_pytorch.py --backend mytorch_jit pytorch --data-root . --train-list splits\temporal_block_gap20\train.txt --val-list splits\temporal_block_gap20\val.txt --epochs 5 --batch-size 32
```

运行 JIT/融合消融：

```powershell
python benchmark_donkeycar_resnet18_jit.py --data-root . --train-list splits\temporal_block_gap20\train.txt --val-list splits\temporal_block_gap20\val.txt --results-dir results\jit_fusion_ablation_full_bs32 --epochs 5 --batch-size 32
```

运行卷积/池化正确性实验：

```powershell
python experiments\conv_pool_correctness.py
```

构建 DonkeyCar Docker 镜像：

```powershell
docker build -f deploy\docker\Dockerfile.donkeycar -t mytorch-donkeycar:latest .
```

运行 DonkeyCar compose：

```powershell
docker compose -f deploy\docker\docker-compose.donkeycar.yml up --build
```

提交 kind Kubernetes Job：

```powershell
kubectl apply -f deploy\k8s\donkeycar-training-job.kind.yaml
kubectl logs -f job/mytorch-donkeycar-train
```

## 14. 当前项目注意事项

| 项目 | 说明 |
|---|---|
| JIT 定位 | 当前不是完整通用 JIT 编译器，更准确是 trace、图优化、模式融合和图执行实验 |
| `auto_drive.py` | 模型构造接口可能落后于当前 `model/resnet.py`，运行前需要改为 `ResNet18Original` 或 `ResNet18` |
| 数据集重复 | `mytorch/dataset.py` 和 `dataset/autodrive_dataset.py` 都实现了 AutoDriveDataset，后者 transform 更灵活 |
| GPU 支持 | MyTorch GPU 依赖 CuPy；没有 CuPy 时自动以 NumPy CPU 路径为主 |
| 训练输出 | 大型输出、权重和缓存被 `.gitignore` 排除，重要结果应单独保留 CSV/Markdown/JSON 或使用 Git LFS |
| DonkeyCar 标签 | 当前训练目标是 steering angle，油门 throttle 通常固定或未训练 |

## 15. 项目框架总结

项目可以分为四层：

| 层级 | 内容 | 代表文件 |
|---|---|---|
| 框架层 | Tensor、自动求导、算子、模块、优化器、数据加载 | `mytorch/tensor.py`、`mytorch/function.py`、`mytorch/modules.py`、`mytorch/optim.py` |
| 模型层 | 自动驾驶 CNN、ResNet18、示例模型 | `model/resnet.py`、`model/autodrive_net.py`、`model/lenet.py` |
| 实验层 | 训练、benchmark、正确性、消融、划分 | `train_donkeycar_resnet18_jit_vs_pytorch.py`、`benchmark_donkeycar_resnet18_jit.py`、`experiments/*.py` |
| 工程化层 | Docker、Kubernetes、Kubeflow、结果管理 | `deploy/**`、`kuflow/**`、`results/**` |

主线是：先在 `mytorch` 中实现深度学习框架能力，再用 `model/resnet.py` 和 DonkeyCar 数据构建自动驾驶回归任务，最后通过实验脚本和部署文件验证框架正确性、性能优化和工程运行能力。
