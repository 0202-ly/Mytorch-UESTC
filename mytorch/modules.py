import numpy as np
import pickle
from .tensor import Tensor
# 引用我们在 function.py 里定义的算子 (确保你的 function.py 中有这些算子)
from .function import (
    Add, MatMul, Conv2dOp, ConvTranspose2dOp, MaxPoolOp, MinPoolOp, AvgPoolOp, ReshapeOp,
    ReLU as ReLU_Op, ELU as ELU_Op, Sigmoid as Sigmoid_Op,
    LogSoftmaxOp, NLLLossOp, BatchNorm2dOp, 
    FusedMSELossOp, FusedBatchNormReLUOp, FusedAddReLUOp, FusedCrossEntropyLossOp, FusedLinearReLUOp,FusedConv2dReLUOp,FusedConvBNReLUOp,
    FusedConvBNAddReLUOp,FusedBatchNormAddReLUOp
)

def _to_2tuple(value):
    """辅助函数：将标量转换为元组，例如 3 -> (3, 3)"""
    if isinstance(value, tuple):
        return value
    return value, value

# ==========================================================
# 核心基类 (终极升级版)
# ==========================================================

class Module:
    def __init__(self):
        self.training = True
    def _apply(self, fn):
        """
        内部辅助函数：递归地对所有子模块和张量应用某个函数 (如 .cuda())
        这是处理 ResNet 这种包含 list/dict 结构的网络的关键！
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, Module):
                attr_value._apply(fn)
            elif isinstance(attr_value, Tensor):
                fn(attr_value)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, Module):
                        item._apply(fn)
                    elif isinstance(item, Tensor):
                        fn(item)
            elif isinstance(attr_value, dict):
                for item in attr_value.values():
                    if isinstance(item, Module):
                        item._apply(fn)
                    elif isinstance(item, Tensor):
                        fn(item)
        return self
    def cuda(self):
        """将模型所有参数转移到 GPU"""
        return self._apply(lambda t: t.cuda())

    def cpu(self):
        """将模型所有参数转移到 CPU"""
        return self._apply(lambda t: t.cpu())

    def train(self):
        """切换到训练模式"""
        self.training = True
        # 递归处理 list 里的子模块
        self._apply_mode(True)
        return self

    def eval(self):
        """切换到评估模式"""
        self.training = False
        self._apply_mode(False)
        return self

    def _apply_mode(self, training):
        """递归切换训练/评估状态的辅助函数"""
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.training = training
                attr._apply_mode(training)

            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        item.training = training
                        item._apply_mode(training)

            elif isinstance(attr, dict):
                for item in attr.values():
                    if isinstance(item, Module):
                        item.training = training
                        item._apply_mode(training)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def zero_grad(self):
        """将模型所有参数的梯度置零"""
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        """
        升级版：递归收集所有参数。
        支持识别存储在 list, tuple 或 dict 中的 Tensor 和 Module。
        使用 seen 集合防止参数被重复收集。
        """
        params = []
        seen = set()

        def collect(obj):
            if isinstance(obj, Module):
                # 遍历模块的所有属性
                for v in obj.__dict__.values():
                    collect(v)
            elif isinstance(obj, Tensor):
                # 只收集需要求导的 Tensor
                if obj.requires_grad:
                    obj_id = id(obj)
                    if obj_id not in seen:
                        seen.add(obj_id)
                        params.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    collect(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    collect(item)

        collect(self)
        return params

    def save_weights(self, path):
        """保存模型权重到文件"""
        params = self.parameters()
        # 提取 Tensor 中的 numpy/cupy 数据并转回 CPU
        weights_data = [p.data.get() if hasattr(p.data, 'get') else p.data for p in params]
        with open(path, 'wb') as f:
            pickle.dump(weights_data, f)
        print(f"模型权重已保存至: {path}")

    def load_weights(self, path):
        """从文件加载模型权重"""
        with open(path, 'rb') as f:
            weights_data = pickle.load(f)

        params = self.parameters()
        if len(weights_data) != len(params):
            raise ValueError("权重文件与模型结构不匹配！")

        for p, d in zip(params, weights_data):
            p.data = p.xp.array(d) if hasattr(p, 'xp') else np.array(d)
        print(f"模型权重已成功从 {path} 加载。")


# ==========================================================
# 线性层
# ==========================================================

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier 初始化
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32),
            requires_grad=True
        )

        self.bias = Tensor(
            np.zeros((1, out_features), dtype=np.float32),
            requires_grad=True
        )
    def forward(self, x):
        return Add()(MatMul()(x, self.weight), self.bias)


# ==========================================================
# 卷积层大家族
# ==========================================================

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels 必须能被 groups 整除")
        if out_channels % groups != 0:
            raise ValueError("out_channels 必须能被 groups 整除")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups

        kh, kw = self.kernel_size
        
        # He 初始化 (针对 ReLU 优化)
        limit = np.sqrt(6 / (in_channels * kh * kw + out_channels))

        # 权重形状: (out_channels, in_channels // groups, k, k)
        self.weight = Tensor(
            np.random.uniform(
                -limit,
                limit,
                (out_channels, in_channels // groups, kh, kw)
            ).astype(np.float32),
            requires_grad=True
        )

        self.bias = Tensor(
            np.zeros((1, out_channels), dtype=np.float32),
            requires_grad=True
        )

    def forward(self, x):
        return Conv2dOp()(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class GroupConv2d(Conv2d):
    """分组卷积语法糖"""
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1, padding=0, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


class DilatedConv2d(Conv2d):
    """空洞卷积语法糖"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=0, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


class DepthwiseSeparableConv2d(Module):
    """深度可分离卷积 (Depthwise Separable Convolution)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, depth_multiplier=1):
        super().__init__()
        mid_channels = in_channels * depth_multiplier
        
        # 1. 逐通道卷积 (Depthwise)
        self.depthwise = Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels
        )
        
        # 2. 逐点卷积 (Pointwise)
        self.pointwise = Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1
        )
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.relu(self.depthwise(x)))


class ConvTranspose2d(Module):
    """转置卷积 (反卷积)"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1
    ):
        super().__init__()

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels 和 out_channels 必须能被 groups 整除")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.output_padding = _to_2tuple(output_padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups

        kh, kw = self.kernel_size

        # 与 Conv2d 保持类似初始化口径
        limit = np.sqrt(6 / (in_channels * kh * kw + out_channels))

        # 转置卷积权重形状：
        # (in_channels, out_channels // groups, kh, kw)
        self.weight = Tensor(
            np.random.uniform(
                -limit,
                limit,
                (in_channels, out_channels // groups, kh, kw)
            ).astype(np.float32),
            requires_grad=True
        )

        # bias 对应输出通道
        self.bias = Tensor(
            np.zeros((1, out_channels), dtype=np.float32),
            requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return ConvTranspose2dOp()(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups
        )
# ==========================================================
# 辅助层
# ==========================================================

class Flatten(Module):
    def forward(self, x):
        # 自动推断 batch_size，拉平后面所有维度
        return ReshapeOp()(x, x.shape()[0], -1)


# ==========================================================
# 激活函数层
# ==========================================================

class ReLU(Module):
    def forward(self, x):
        return ReLU_Op()(x)
class Identity(Module):
    def forward(self, x):
        return x
class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return ELU_Op(self.alpha)(x)

class Sigmoid(Module):
    def forward(self, x):
        return Sigmoid_Op()(x)


# ==========================================================
# 池化层
# ==========================================================

class MaxPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_2tuple(kernel_size)
        # 如果没有指定 stride，默认等于 kernel_size
        self.stride = _to_2tuple(stride if stride is not None else kernel_size)
        self.padding = _to_2tuple(padding)

    def forward(self, x):
        return MaxPoolOp()(x, self.kernel_size, self.stride, self.padding)


class AvgPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride if stride is not None else kernel_size)
        self.padding = _to_2tuple(padding)

    def forward(self, x):
        return AvgPoolOp()(x, self.kernel_size, self.stride, self.padding)


class MinPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride if stride is not None else kernel_size)
        self.padding = _to_2tuple(padding)

    def forward(self, x):
        return MinPoolOp()(x, self.kernel_size, self.stride, self.padding)

# mytorch/modules.py

class AdaptiveAvgPool2d(Module):
    """
    自适应平均池化
    根据输入形状自动计算池化参数，确保输出形状固定
    """
    def __init__(self, output_size):
        super().__init__()
        # 统一转换为 (H, W) 元组，支持单整数输入 
        self.output_size = _to_2tuple(output_size)

    def forward(self, x):
        # 获取输入张量的形状 (N, C, H, W)
        _, _, ih, iw = x.shape()
        oh, ow = self.output_size

        # 计算步长和卷积核大小
        stride_h = ih // oh
        stride_w = iw // ow
        
        kernel_h = ih - (oh - 1) * stride_h
        kernel_w = iw - (ow - 1) * stride_w

        # 调用已有的 AvgPoolOp 算子执行计算
        # 这样可以确保操作被记录在计算图中，支持自动反向传播 [cite: 2, 3]
        return AvgPoolOp()(x, (kernel_h, kernel_w), (stride_h, stride_w), padding=0)

# ==========================================================
# 归一化层
# ==========================================================

class BatchNorm2d(Module):
    """
    二维批量归一化层 (主要配合卷积层使用)
    在组装 ResNet 等现代视觉网络时是必备组件。
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 将权重 (gamma) 和偏置 (beta) 用 Tensor 包装，形状设为 (1, C, 1, 1)，以支持 4D 张量广播计算。
        self.weight = Tensor(
            np.ones((1, num_features, 1, 1), dtype=np.float32),
            requires_grad=True
        )

        self.bias = Tensor(
            np.zeros((1, num_features, 1, 1), dtype=np.float32),
            requires_grad=True
        )

        self.running_mean = Tensor(
            np.zeros((1, num_features, 1, 1), dtype=np.float32),
            requires_grad=False
        )

        self.running_var = Tensor(
            np.ones((1, num_features, 1, 1), dtype=np.float32),
            requires_grad=False
        )
    def forward(self, x):
        # 将当前的 training 模式传入底层 Op 中，如果在 eval 模式下会自动使用全局 mean/var
        return BatchNorm2dOp(momentum=self.momentum, eps=self.eps, is_train=self.training)(
            x, self.weight, self.bias, self.running_mean, self.running_var
        )

class FusedBatchNormReLU(Module):
    """
    训练阶段使用的 BatchNorm2d + ReLU 融合模块。

    它复用原 BatchNorm2d 的参数和 running_mean/running_var。
    """
    def __init__(self, bn: BatchNorm2d):
        super().__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum

        # 直接复用原 BN 的 Tensor，保证 optimizer 仍然能更新同一份参数
        self.weight = bn.weight
        self.bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var

        self.training = bn.training

    def forward(self, x):
        return FusedBatchNormReLUOp(
            momentum=self.momentum,
            eps=self.eps,
            is_train=self.training
        )(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var
        )

class FusedAddReLU(Module):
    """
    Module 包装版 Add + ReLU。

    用法：
        out = self.add_relu(out, identity)
    """

    def forward(self, x1, x2):
        return FusedAddReLUOp()(x1, x2)

class FusedLinearReLU(Module):
    """
    Linear + ReLU 融合模块。

    用法:
        self.fc_relu = FusedLinearReLU(in_features, out_features)

    或者从已有 Linear 创建:
        self.fc_relu = FusedLinearReLU.from_linear(old_linear)
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        limit = np.sqrt(6 / (in_features + out_features))

        self.weight = Tensor(
            np.random.uniform(
                -limit,
                limit,
                (in_features, out_features)
            ).astype(np.float32),
            requires_grad=True
        )

        self.bias = Tensor(
            np.zeros((1, out_features), dtype=np.float32),
            requires_grad=True
        )

    @classmethod
    def from_linear(cls, linear):
        """
        复用已有 Linear 的 weight / bias。
        这样 optimizer 仍然能拿到同一份参数 Tensor。
        """
        in_features = linear.weight.shape()[0]
        out_features = linear.weight.shape()[1]

        obj = cls(in_features, out_features)
        obj.weight = linear.weight
        obj.bias = linear.bias

        return obj

    def forward(self, x):
        return FusedLinearReLUOp()(x, self.weight, self.bias)
    
class FusedBatchNormAddReLU(Module):
    """
    训练态 BatchNorm2d + residual Add + ReLU 融合模块。

    用于 ResNet BasicBlock:
        conv2 -> bn2 -> add(identity) -> relu2

    它复用原 BatchNorm2d 的 weight / bias / running_mean / running_var。
    """
    def __init__(self, bn: BatchNorm2d):
        super().__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum

        self.weight = bn.weight
        self.bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var

        self.training = bn.training

    def forward(self, x, identity):
        return FusedBatchNormAddReLUOp(
            momentum=self.momentum,
            eps=self.eps,
            is_train=self.training
        )(
            x,
            identity,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var
        )
    
class FusedConv2dReLU(Conv2d):
    """
    Module 包装版 Conv2d + ReLU 静态融合层。

    注意：
    - 它继承 Conv2d，复用原来的 weight / bias。
    - forward 调用 FusedConv2dReLUOp。
    - 内部卷积仍然使用 im2col + einsum。
    """

    @classmethod
    def from_conv2d(cls, conv):
        obj = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups
        )

        obj.weight = conv.weight
        obj.bias = conv.bias

        return obj

    def forward(self, x):
        return FusedConv2dReLUOp()(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
    
class FusedConvBNReLU(Module):
    """
    Module 包装版 Conv2d + BatchNorm2d + ReLU。

    复用原 Conv2d 的 weight/bias。
    复用原 BatchNorm2d 的 gamma/beta/running_mean/running_var。
    """

    def __init__(self, conv: Conv2d, bn: BatchNorm2d):
        super().__init__()

        if not isinstance(conv, Conv2d):
            raise TypeError(f"conv must be Conv2d, got {type(conv)}")

        if not isinstance(bn, BatchNorm2d):
            raise TypeError(f"bn must be BatchNorm2d, got {type(bn)}")

        if conv.out_channels != bn.num_features:
            raise ValueError(
                f"conv.out_channels must equal bn.num_features, "
                f"got {conv.out_channels} vs {bn.num_features}"
            )

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = conv.weight
        self.bias = conv.bias

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum

        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var

        self.training = bn.training

    def forward(self, x):
        return FusedConvBNReLUOp(
            momentum=self.momentum,
            eps=self.eps,
            is_train=self.training,
        )(
            x,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class FusedConvBNAddReLU(Module):
    """
    Module 包装版 Conv2d + BatchNorm2d + Add(identity) + ReLU。

    用于 ResNet BasicBlock 第二个分支：
        conv2 -> bn2 -> add(identity) -> relu2
    """

    def __init__(self, conv: Conv2d, bn: BatchNorm2d):
        super().__init__()

        if not isinstance(conv, Conv2d):
            raise TypeError(f"conv must be Conv2d, got {type(conv)}")

        if not isinstance(bn, BatchNorm2d):
            raise TypeError(f"bn must be BatchNorm2d, got {type(bn)}")

        if conv.out_channels != bn.num_features:
            raise ValueError(
                f"conv.out_channels must equal bn.num_features, "
                f"got {conv.out_channels} vs {bn.num_features}"
            )

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = conv.weight
        self.bias = conv.bias

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum

        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var

        self.training = bn.training

    def forward(self, x, identity):
        return FusedConvBNAddReLUOp(
            momentum=self.momentum,
            eps=self.eps,
            is_train=self.training,
        )(
            x,
            identity,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )