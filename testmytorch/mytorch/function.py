from .tensor import Tensor
import numpy as np

# 尝试导入 CuPy，如果不存在则回退到 NumPy
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


# =========================================================================
# 辅助函数 (Im2Col / Col2Im) - 适配 CPU/GPU
# =========================================================================

def _im2col(input_data, kernel_h, kernel_w, stride=1, padding=0, xp=np):
    """
    将 N*C*H*W 的输入数据转换为 Im2Col 矩阵。
    增加 xp 参数以支持 cupy。
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    # Padding
    img = xp.pad(input_data, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')

    # 初始化 col 矩阵
    col = xp.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=input_data.dtype)

    # 填充 col (使用切片，cupy 支持这种操作)
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # Reshape
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col, H, W


def _col2im(col, input_shape, kernel_h, kernel_w, stride=1, padding=0, xp=np):
    """
    将 Im2Col 矩阵转换回 N*C*H*W 的梯度形状。
    增加 xp 参数以支持 cupy。
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)

    img = xp.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1), dtype=col.dtype)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding:H + padding, padding:W + padding]


# =========================================================================
# Function 基类
# =========================================================================

class Function:
    """自动求导操作的抽象基类。"""

    def __init__(self):
        self.data = None  # 存储前向传播的输出
        self.grad = None  # 存储反向传播的上游梯度
        self.inputs = []  # 存储输入 Tensor
        self.xp = np  # 当前计算后端 (np 或 cp)

    def __call__(self, *args, **kwargs):
        # 1. 保存输入以便 backward 使用
        self.inputs = args

        # 2. 自动判定计算后端 (Backend)
        # 如果输入中有 Tensor 且使用了 CuPy，则整个 Op 使用 CuPy
        if len(args) > 0 and hasattr(args[0], 'xp'):
            self.xp = args[0].xp
        else:
            self.xp = np

        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad=None):
        raise NotImplementedError

    def _get_inputs(self) -> list:
        raise NotImplementedError


# =========================================================================
# 基础算子 (Add, MatMul)
# =========================================================================

class Add(Function):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None

    def forward(self, a: Tensor, b: Tensor):
        self.a = a
        self.b = b
        # 使用 a.data 和 b.data 进行加法，NumPy/CuPy 会自动处理
        self.data = self.a.data + self.b.data

        requires_grad = a.requires_grad or b.requires_grad
        output_tensor = Tensor(self.data, requires_grad=requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        # 如果由 Tensor 调用 backward，grad 会自动传入 self.grad
        # 但为了兼容性，这里处理显式传参情况
        if grad is not None: self.grad = grad

        if self.a.requires_grad:
            if self.a.grad is None: self.a.grad = self.xp.zeros_like(self.a.data)
            grad_a = self.grad
            # 检查广播: 使用 self.xp 进行比较
            if self.grad.shape != self.a.shape():
                axes_to_sum = tuple(i for i, (s_grad, s_a) in enumerate(zip(self.grad.shape, self.a.shape())) if
                                    s_a == 1 and s_grad > 1)
                if axes_to_sum:
                    grad_a = grad_a.sum(axis=axes_to_sum, keepdims=True)
            self.a.grad += grad_a

        if self.b.requires_grad:
            if self.b.grad is None: self.b.grad = self.xp.zeros_like(self.b.data)
            grad_b = self.grad
            if self.grad.shape != self.b.shape():
                axes_to_sum = tuple(i for i, (s_grad, s_b) in enumerate(zip(self.grad.shape, self.b.shape())) if
                                    s_b == 1 and s_grad > 1)
                if axes_to_sum:
                    grad_b = grad_b.sum(axis=axes_to_sum, keepdims=True)
            self.b.grad += grad_b

    def _get_inputs(self):
        return [self.a, self.b]


class MatMul(Function):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None

    def forward(self, a: Tensor, b: Tensor):
        self.a = a
        self.b = b
        # 使用 self.xp.dot
        self.data = self.xp.dot(self.a.data, self.b.data)

        requires_grad = a.requires_grad or b.requires_grad
        output_tensor = Tensor(self.data, requires_grad=requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad

        if self.a.requires_grad:
            if self.a.grad is None: self.a.grad = self.xp.zeros_like(self.a.data)
            self.a.grad += self.xp.dot(self.grad, self.b.data.T)
        if self.b.requires_grad:
            if self.b.grad is None: self.b.grad = self.xp.zeros_like(self.b.data)
            self.b.grad += self.xp.dot(self.a.data.T, self.grad)

    def _get_inputs(self):
        return [self.a, self.b]


# =========================================================================
# 激活函数 (ReLU, ELU, Sigmoid)
# =========================================================================

class ReLU(Function):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x: Tensor):
        self.x = x
        # self.xp.where 或 self.xp.maximum
        self.data = self.xp.where(x.data > 0, x.data, 0.0)
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad
        if self.grad is None: return

        grad_mask = self.xp.where(self.x.data > 0, 1.0, 0.0)
        grad_to_prev = self.grad * grad_mask

        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(grad_to_prev)
            self.x.grad += grad_to_prev

    def _get_inputs(self):
        return [self.x]


class ELU(Function):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.x = None

    def forward(self, x: Tensor):
        self.x = x
        # self.xp.exp
        self.data = self.xp.where(
            x.data > 0,
            x.data,
            self.alpha * (self.xp.exp(x.data) - 1)
        )
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad
        if self.grad is None: return

        if self.x.requires_grad:
            grad_local = self.xp.where(
                self.x.data > 0,
                1.0,
                self.alpha * self.xp.exp(self.x.data)
            )
            grad_to_pass = self.grad * grad_local

            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += grad_to_pass

    def _get_inputs(self):
        return [self.x]


class Sigmoid(Function):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x: Tensor):
        self.x = x
        self.data = 1.0 / (1.0 + self.xp.exp(-x.data))
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad

        grad_local = self.data * (1.0 - self.data)
        grad_to_pass = self.grad * grad_local

        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += grad_to_pass

    def _get_inputs(self):
        return [self.x]


# =========================================================================
# 卷积与池化 (Conv2d, MaxPool, MinPool, AvgPool)
# =========================================================================

class Conv2dOp(Function):
    def __init__(self):
        super().__init__()
        self.x = None
        self.w = None
        self.b = None
        self.stride = None
        self.padding = None
        self.col = None

    def forward(self, x: Tensor, w: Tensor, b: Tensor = None, stride: int = 1, padding: int = 0):
        self.x = x
        self.w = w
        self.b = b
        self.stride = stride
        self.padding = padding
        self.x_shape = x.shape()

        FN, C, KH, KW = w.shape()
        N, C, H, W = x.shape()

        # 传递 self.xp 给 im2col
        col, _, _ = _im2col(x.data, KH, KW, stride, padding, xp=self.xp)
        self.col = col

        W_reshaped = w.data.reshape(FN, -1).T
        out_flat = self.xp.dot(col, W_reshaped)

        if b is not None:
            out_flat += b.data

        out_h = (H + 2 * padding - KH) // stride + 1
        out_w = (W + 2 * padding - KW) // stride + 1
        self.data = out_flat.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        requires_grad = x.requires_grad or w.requires_grad or (b and b.requires_grad)
        output_tensor = Tensor(self.data, requires_grad=requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad
        dL_dO = self.grad

        FN, C, KH, KW = self.w.shape()
        N, _, H, W = self.x_shape

        dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(-1, FN)

        if self.b is not None and self.b.requires_grad:
            dL_dB = self.xp.sum(dL_dO, axis=(0, 2, 3))
            if self.b.grad is None: self.b.grad = self.xp.zeros_like(dL_dB)
            self.b.grad += dL_dB.reshape(self.b.shape())

        if self.w.requires_grad:
            dL_dW_flat = self.xp.dot(self.col.T, dL_dO_flat)
            dL_dW = dL_dW_flat.T.reshape(FN, C, KH, KW)
            if self.w.grad is None: self.w.grad = self.xp.zeros_like(dL_dW)
            self.w.grad += dL_dW

        if self.x.requires_grad:
            W_reshaped = self.w.data.reshape(FN, -1).T
            dL_dCol = self.xp.dot(dL_dO_flat, W_reshaped.T)

            # 传递 self.xp 给 col2im
            dL_dX = _col2im(dL_dCol, self.x_shape, KH, KW, self.stride, self.padding, xp=self.xp)

            if self.x.grad is None: self.x.grad = self.xp.zeros_like(dL_dX)
            self.x.grad += dL_dX

    def _get_inputs(self):
        parents = [self.x, self.w]
        if self.b is not None:
            parents.append(self.b)
        return parents


class MaxPoolOp(Function):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x: Tensor, kernel_size, stride, padding=0):
        self.x = x
        self.x_shape = x.shape()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        N, C, H, W = self.x_shape
        KH, KW = kernel_size, kernel_size

        out_h = (H + 2 * self.padding - KH) // stride + 1
        out_w = (W + 2 * self.padding - KW) // stride + 1

        col, _, _ = _im2col(x.data, KH, KW, stride=self.stride, padding=self.padding, xp=self.xp)
        col_reshaped = col.reshape(-1, C, KH * KW)

        self.data = self.xp.max(col_reshaped, axis=2)
        self.mask = self.xp.argmax(col_reshaped, axis=2)

        self.data = self.data.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad

        if self.x.requires_grad:
            dL_dO = self.grad
            N, C, Out_H, Out_W = dL_dO.shape
            dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(N * Out_H * Out_W, C)
            KH, KW = self.kernel_size, self.kernel_size

            dL_dCol = self.xp.zeros((dL_dO_flat.shape[0], C * KH * KW), dtype=dL_dO.dtype)

            idx_range = self.xp.arange(dL_dCol.shape[0])[:, None]
            c_range = self.xp.arange(C)[None, :]
            target_indices = c_range * (KH * KW) + self.mask

            # Scatter
            dL_dCol[idx_range, target_indices] = dL_dO_flat

            dL_dX = _col2im(dL_dCol, self.x_shape, KH, KW, self.stride, padding=self.padding, xp=self.xp)

            if self.x.grad is None: self.x.grad = self.xp.zeros_like(dL_dX)
            self.x.grad += dL_dX

    def _get_inputs(self):
        return [self.x]


class MinPoolOp(Function):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x: Tensor, kernel_size, stride, padding=0):
        self.x = x
        self.x_shape = x.shape()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        N, C, H, W = self.x_shape
        KH, KW = kernel_size, kernel_size

        out_h = (H + 2 * self.padding - KH) // stride + 1
        out_w = (W + 2 * self.padding - KW) // stride + 1

        col, _, _ = _im2col(x.data, KH, KW, stride=self.stride, padding=self.padding, xp=self.xp)
        col_reshaped = col.reshape(-1, C, KH * KW)

        self.data = self.xp.min(col_reshaped, axis=2)
        self.mask = self.xp.argmin(col_reshaped, axis=2)

        self.data = self.data.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad
        if self.x.requires_grad:
            dL_dO = self.grad
            N, C, Out_H, Out_W = dL_dO.shape
            dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(N * Out_H * Out_W, C)
            KH, KW = self.kernel_size, self.kernel_size

            dL_dCol = self.xp.zeros((dL_dO_flat.shape[0], C * KH * KW), dtype=dL_dO.dtype)

            idx_range = self.xp.arange(dL_dCol.shape[0])[:, None]
            c_range = self.xp.arange(C)[None, :]
            target_indices = c_range * (KH * KW) + self.mask

            # Scatter
            dL_dCol[idx_range, target_indices] = dL_dO_flat  # 修正：这里不需要 [:, c_range]，因为 dL_dO_flat 已经是 (Row, C)

            dL_dX = _col2im(dL_dCol, self.x_shape, KH, KW, self.stride, padding=self.padding, xp=self.xp)

            if self.x.grad is None: self.x.grad = self.xp.zeros_like(dL_dX)
            self.x.grad += dL_dX

    def _get_inputs(self):
        return [self.x]


class AvgPoolOp(Function):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, kernel_size, stride, padding=0):
        self.x = x
        self.x_shape = x.shape()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        KH, KW = kernel_size, kernel_size
        self.pool_area = KH * KW

        col, _, _ = _im2col(x.data, KH, KW, stride=stride, padding=self.padding, xp=self.xp)

        self.data = self.xp.mean(col.reshape(-1, x.shape()[1], self.pool_area), axis=2)

        N, C, H, W = x.shape()
        out_h = (H + 2 * self.padding - KH) // stride + 1
        out_w = (W + 2 * self.padding - KW) // stride + 1
        self.data = self.data.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad
        if self.x.requires_grad:
            dL_dO = self.grad
            N, C, Out_H, Out_W = dL_dO.shape
            dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(N * Out_H * Out_W, C)

            KH, KW = self.kernel_size, self.kernel_size

            dL_dO_distributed = dL_dO_flat / self.pool_area
            # 使用 self.xp.repeat
            dL_dCol = self.xp.repeat(dL_dO_distributed, self.pool_area, axis=1)

            dL_dX = _col2im(dL_dCol, self.x_shape, KH, KW, self.stride, padding=self.padding, xp=self.xp)

            if self.x.grad is None: self.x.grad = self.xp.zeros_like(dL_dX)
            self.x.grad += dL_dX

    def _get_inputs(self):
        return [self.x]


# =========================================================================
# Reshape & Loss Functions
# =========================================================================

class ReshapeOp(Function):
    def __init__(self):
        super().__init__()
        self.x_shape = None

    def forward(self, x: Tensor, *shape):
        self.x = x
        self.x_shape = x.shape()
        self.data = x.data.reshape(*shape)

        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad
        if self.x.requires_grad:
            grad_to_pass = self.grad.reshape(self.x_shape)
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(grad_to_pass)
            self.x.grad += grad_to_pass

    def _get_inputs(self):
        return [self.x]


class MSE(Function):
    def __init__(self):
        super().__init__()

    def forward(self, y_prediction: Tensor, y_true: Tensor):
        self.y_prediction = y_prediction
        self.y_true = y_true

        error = self.y_prediction.data - self.y_true.data
        squared_error = error ** 2
        self.data = self.xp.mean(squared_error)

        requires_grad = y_prediction.requires_grad
        output_tensor = Tensor(self.data, requires_grad=requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad

        y_true_data = self.y_true.data.reshape(self.y_prediction.data.shape)
        N = self.y_prediction.data.size

        grad_local = (2.0 / N) * (self.y_prediction.data - y_true_data)
        grad_to_pass = self.grad * grad_local

        if self.y_prediction.requires_grad:
            if self.y_prediction.grad is None:
                self.y_prediction.grad = self.xp.zeros_like(self.y_prediction.data)
            self.y_prediction.grad += grad_to_pass

    def _get_inputs(self):
        return [self.y_prediction, self.y_true]


class LogSoftmaxOp(Function):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        self.x = x

        # 使用 self.xp
        max_x = self.x.data.max(axis=1, keepdims=True)
        stable_x = self.x.data - max_x
        exp_stable_x = self.xp.exp(stable_x)
        sum_exp = exp_stable_x.sum(axis=1, keepdims=True)
        log_sum_exp = self.xp.log(sum_exp)

        self.data = stable_x - log_sum_exp
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad

        if self.x.requires_grad:
            softmax = self.xp.exp(self.data)
            sum_grad = self.grad.sum(axis=1, keepdims=True)
            grad_to_pass = self.grad - (softmax * sum_grad)

            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += grad_to_pass

    def _get_inputs(self):
        return [self.x]


class NLLLossOp(Function):
    def __init__(self):
        super().__init__()

    def forward(self, log_probs: Tensor, target: Tensor):
        self.log_probs = log_probs
        self.target = target
        self.N, self.C = self.log_probs.shape()

        # 使用 self.xp.arange
        # 注意: target.data 如果是 cupy array, 必须用 cupy 的高级索引
        picked_log_probs = self.log_probs.data[self.xp.arange(self.N), self.target.data]

        self.data = -self.xp.mean(picked_log_probs)

        output_tensor = Tensor(self.data, requires_grad=self.log_probs.requires_grad, creator=self)
        return output_tensor

    def backward(self, grad=None):
        if grad is not None: self.grad = grad

        if self.log_probs.requires_grad:
            grad_to_pass = self.xp.zeros_like(self.log_probs.data)

            # 使用 self.xp.arange
            grad_to_pass[self.xp.arange(self.N), self.target.data] = -1.0 / self.N
            grad_to_pass *= self.grad

            if self.log_probs.grad is None:
                self.log_probs.grad = self.xp.zeros_like(self.log_probs.data)
            self.log_probs.grad += grad_to_pass

    def _get_inputs(self):
        return [self.log_probs, self.target]