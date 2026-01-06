from .function import Sigmoid as Sigmoid_Op
from .function import ReLU as ReLU_Op
from .function import Add, MatMul, Conv2dOp, MaxPoolOp, MinPoolOp, AvgPoolOp, ReshapeOp
from .tensor import Tensor


class Module:
    """
    所有神经网络模块的基类。
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        """
        允许像调用函数一样调用模块实例，直接触发 forward 方法。
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        定义每次前向传播的计算逻辑。
        必须在子类中实现。
        """
        raise NotImplementedError("You must implement the 'forward' method!")

    def backward(self):
        """
        模块的反向传播通常由 Autograd 引擎（通过 Tensor 和 Op）自动处理。
        此方法通常预留给需要手动控制梯度的特殊模块。
        """
        pass

    def parameters(self):
        """
        返回该模块中所有需要优化的参数（Tensor 列表）。
        """
        raise NotImplementedError("You must collect all the parameters!")


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # 关键：每次 forward 时实例化一个新的 Op
        # Op 对象是有状态的（保存了反向传播所需的中间值，如 input 或 mask），
        # 因此每次前向传播必须创建独立的计算节点。
        return ReLU_Op()(x)

    def parameters(self):
        # 激活函数没有可学习参数
        return []


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return Sigmoid_Op()(x)

    def parameters(self):
        return []


class Linear(Module):
    """
    全连接层 (Fully Connected Layer)
    执行 y = x @ W + b
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        # 初始化权重: (In, Out)
        self.weights = Tensor.random_matrix(input_size, output_size, requires_grad=True)
        # 初始化偏置: (1, Out) - 支持广播
        self.bias = Tensor.random_matrix(1, output_size, requires_grad=True)

    def forward(self, x: Tensor):
        # 1. 线性变换: X (N, In) @ W (In, Out) -> (N, Out)
        affine = MatMul()(x, self.weights)

        # 2. 加上偏置: (N, Out) + (1, Out) -> (N, Out)
        output = Add()(affine, self.bias)

        return output

    def parameters(self):
        return [self.weights, self.bias]


class Conv2d(Module):
    """
    二维卷积层
    输入形状: (N, C_in, H, W)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 权重形状: (Out_Channels, In_Channels, Kernel_H, Kernel_W)
        self.weights = Tensor.random_matrix(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)

        # 偏置形状: (1, Out_Channels)
        # 注意：在 Conv2dOp 中，偏置会被广播并加到对应的通道上
        self.bias = Tensor.random_matrix(1, out_channels, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # 调用卷积 Op
        return Conv2dOp()(x, self.weights, self.bias, self.stride, self.padding)

    def parameters(self):
        return [self.weights, self.bias]


class MaxPool(Module):
    """
    最大池化层
    """

    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return MaxPoolOp()(x, self.kernel_size, self.stride, self.padding)

    def parameters(self):
        return []


class AvgPool(Module):
    """
    平均池化层
    """

    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return AvgPoolOp()(x, self.kernel_size, self.stride, self.padding)

    def parameters(self):
        return []


class MinPool(Module):
    """
    最小池化层
    """

    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return MinPoolOp()(x, self.kernel_size, self.stride, self.padding)

    def parameters(self):
        return []


class Flatten(Module):
    """
    展平层
    通常用于卷积层到全连接层的过渡。
    将 (N, C, H, W) 展平为 (N, C*H*W)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape()[0]
        # 使用 -1 让 NumPy 自动计算剩余维度的总大小
        return ReshapeOp()(x, batch_size, -1)

    def parameters(self):
        return []





