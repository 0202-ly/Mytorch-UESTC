import numpy as np
from .tensor import Tensor
# 引用我们在 function.py 里定义的算子
from .function import (
    Add, MatMul, Conv2dOp, MaxPoolOp, MinPoolOp, AvgPoolOp, ReshapeOp,
    ReLU as ReLU_Op, ELU as ELU_Op, Sigmoid as Sigmoid_Op,
    LogSoftmaxOp, NLLLossOp
)


class Module:
    def __init__(self):
        self.training = True

    def train(self):
        """切换到训练模式"""
        self.training = True
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.train()

    def eval(self):
        """切换到评估模式"""
        self.training = False
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.eval()

    def cuda(self):
        """将模型内所有参数移动到 GPU"""
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.cuda()
            elif hasattr(attr, 'cuda'):  # 针对 Parameter (Tensor)
                attr.cuda()
        return self

    def cpu(self):
        """将模型移回 CPU"""
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.cpu()
            elif hasattr(attr, 'cpu'):
                attr.cpu()
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        """递归收集所有子模块的参数"""
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
        return params


# ==========================================================
# 常用层实现
# ==========================================================

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier 初始化
        limit = np.sqrt(6 / (in_features + out_features))
        self.w = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)),
            requires_grad=True
        )
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x):
        # Linear = MatMul(x, w) + b
        return Add()(MatMul()(x, self.w), self.b)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He 初始化 (针对 ReLU)
        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels))

        # 权重形状: (out_channels, in_channels, k, k)
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros((1, out_channels)), requires_grad=True)

    def forward(self, x):
        return Conv2dOp()(x, self.weight, self.bias, self.stride, self.padding)


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
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return MaxPoolOp()(x, self.kernel_size, self.stride, self.padding)


class AvgPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return AvgPoolOp()(x, self.kernel_size, self.stride, self.padding)


class MinPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return MinPoolOp()(x, self.kernel_size, self.stride, self.padding)

# ==========================================================
# Dropout (适配 GPU版