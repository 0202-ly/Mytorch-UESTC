from .function import Sigmoid as Sigmoid_Op
from .function import ReLU as ReLU_Op
from .function import Add, MatMul,Conv2dOp, MaxPoolOp,MinPoolOp, AvgPoolOp,ReshapeOp
from .tensor import Tensor


class Module:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self,*args, **kwargs):
        raise NotImplementedError("You must implement the 'forward'!")

    def backward(self):
        pass

    def parameters(self):
        raise NotImplementedError("You must collect all the parameters!")

# ... (在 ReLU, Flatten, Sequential 等模块之后)
class ReLU(Module):
    def __init__(self):
        super().__init__()
        # 创建一个 Op 实例
        self.op = ReLU_Op()

    def forward(self, x: Tensor) -> Tensor:
        # 调用 Op 的 forward
        return self.op(x)

    def parameters(self):
        return []
class Sigmoid(Module):
    """
    将 Sigmoid 操作 (Op) 封装为一个模块 (Module)
    """
    def __init__(self):
        super().__init__()
        # 创建一个持久化的 Op 实例
        self.op = Sigmoid_Op()

    def forward(self, x: Tensor) -> Tensor:
        # 调用 Op 的 forward
        return self.op(x)

    def parameters(self):
        # Sigmoid 没有参数
        return []
class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = Tensor.random_matrix( input_size, output_size,requires_grad=True)
        self.bias = Tensor.random_matrix( 1,output_size,requires_grad=True)
        self.matmul_op = MatMul()
        self.add_op = Add()
    '''def forward(self, x:Tensor):
        return Tensor(np.matmul(x.data, self.weights.data) + self.bias.data)'''

    def forward(self, x: Tensor):
        # --- 修正：使用 Ops 构建计算图 ---
        # 1. Y = X @ W
        affine = self.matmul_op(x, self.weights)
        # 2. Out = Y + b (Add Op 会自动处理广播)
        output = self.add_op(affine, self.bias)
        return output
    def backward(self):
        pass

    def parameters(self):
        return [self.weights, self.bias]

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # --- 修正：使用已修复的 random_matrix 创建 4D 权重和 2D 偏置 ---
        self.weights = Tensor.random_matrix(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)
        self.bias = Tensor.random_matrix(1, out_channels, requires_grad=True)

        # --- 修正：创建 Conv2dOp 实例 ---
        self.op = Conv2dOp()  #

    def forward(self, x : Tensor) -> Tensor:
        # --- 修正：调用 Op 的 forward，而不是手动循环 ---
        return self.op(x, self.weights, self.bias, self.stride, self.padding)

    def backward(self):
        pass # Op 负责 Autograd

    def parameters(self):
        # --- 修正：返回 weights 和 bias ---
        return [self.weights, self.bias]


class MaxPool(Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.op = MaxPoolOp()

    def forward(self, x: Tensor) -> Tensor:
        # --- 修正：调用 Op 的 forward ---
        return self.op(x, self.kernel_size, self.stride, self.padding)

    def backward(self):
        pass  # Op 负责 Autograd

    def parameters(self):
        return []

class AvgPool(Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.op = AvgPoolOp()


    def forward(self, x : Tensor) -> Tensor:
        # --- 修正：调用 Op 的 forward ---
        return self.op(x, self.kernel_size, self.stride,self.padding)

    def backward(self):
        pass # Op 负责 Autograd

    def parameters(self):
        return []


class MinPool(Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.op = MinPoolOp()

    def forward(self, x : Tensor) -> Tensor:
        # --- 修正：必须传递 padding ---
        return self.op(x, self.kernel_size, self.stride, self.padding)

    def backward(self):
        pass

    def parameters(self):
        return []


class Flatten(Module):
    def __init__(self):
        super().__init__()
        # --- 修正：创建 ReshapeOp 实例 ---
        self.op = ReshapeOp()

    def forward(self, x: Tensor) -> Tensor:
        # --- 修正：调用 Op 来构建计算图 ---
        batch_size = x.shape()[0]
        # 展平为 (N, C*H*W)
        return self.op(x, batch_size, -1)

    def backward(self):
        pass  # Op 负责 Autograd

    def parameters(self):
        return []

# class BatchNorm2d(Module):
#     def __init__(self, num_features: int, eps: float = 1e-5,
#                  device=None, dtype=None) -> None:
#         super().__init__()
#         self.num_features = num_features
#
#     def forward(self, x:Tensor) -> Tensor:




