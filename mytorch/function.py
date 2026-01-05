from .tensor import Tensor
import numpy as np


class Function:
    """自动求导操作的抽象基类。"""

    def __init__(self):
        self.data = None  # 存储前向传播的输出
        self.grad = None  # 存储反向传播的上游梯度
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """执行前向计算。"""
        raise NotImplementedError

    def backward(self):
        """计算并反向传播梯度。"""
        raise NotImplementedError

    def _get_inputs(self) -> list:
        """返回此操作所依赖的输入 Tensor 列表，用于构建计算图。"""
        raise NotImplementedError
class Add(Function):
    def __init__(self):
        super().__init__()
        # a && b is Tensor
        self.a = None
        self.b = None
        self.data = None
        self.grad = None

    def forward(self, a: Tensor, b: Tensor):
        self.a = a
        self.b = b
        self.data = self.a.data + self.b.data

        # 确定输出是否需要梯度
        requires_grad = a.requires_grad or b.requires_grad
        # 创建输出 Tensor，并将其 creator 设置为 self (这个 Add 实例)
        output_tensor = Tensor(self.data, requires_grad=requires_grad, creator=self)
        return output_tensor

    def backward(self):
        # 处理广播 (e.g., data (N,C) + bias (1,C))
        if self.a.requires_grad:
            if self.a.grad is None: self.a.grad = np.zeros_like(self.a.data)
            grad_a = self.grad
            # 检查广播
            if self.grad.shape != self.a.shape():
                # 维度不同，说明 self.a 被广播了，梯度需要求和
                axes_to_sum = tuple(i for i, (s_grad, s_a) in enumerate(zip(self.grad.shape, self.a.shape())) if
                                    s_a == 1 and s_grad > 1)
                if axes_to_sum:
                    grad_a = grad_a.sum(axis=axes_to_sum, keepdims=True)
            self.a.grad += grad_a
        if self.b.requires_grad:
            if self.b.grad is None: self.b.grad = np.zeros_like(self.b.data)
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
        self.data = None
        self.grad = None


    def forward(self, a: Tensor, b: Tensor):
        self.a = a
        self.b = b
        self.data = np.dot(self.a.data, self.b.data)

        requires_grad = a.requires_grad or b.requires_grad
        output_tensor = Tensor(self.data, requires_grad=requires_grad, creator=self)
        return output_tensor
    def backward(self):
        if self.a.requires_grad:
            if self.a.grad is None: self.a.grad = np.zeros_like(self.a.data)
            self.a.grad += np.dot(self.grad, self.b.data.T)
        if self.b.requires_grad:
            if self.b.grad is None: self.b.grad = np.zeros_like(self.b.data)
            self.b.grad += np.dot(self.a.data.T, self.grad)

    def _get_inputs(self):
        return [self.a, self.b]

class ReLU(Function):
    def __init__(self):
        super().__init__()
        self.x = None
        self.data = None
        self.grad = None

    def forward(self, x: Tensor):
        self.x = x
        self.data = np.where(x.data > 0, x.data, 0.0)
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad)
        output_tensor.creator = self
        return output_tensor

    def backward(self):
        if self.grad is None:
            raise ValueError("ReLU 反向传播前需先执行前向传播")
        grad_mask = np.where(self.x.data > 0, 1.0, 0.0)
        grad_to_prev = self.grad * grad_mask
        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = np.zeros_like(grad_to_prev)
            self.x.grad += grad_to_prev

    def _get_inputs(self):
        return [self.x]

class Sigmoid(Function):
    """
    Sigmoid 激活函数 Op
    """
    def __init__(self):
        super().__init__()
        self.x = None
        self.data = None  # 将存储 sigmoid(x) 的结果
        self.grad = None

    def forward(self, x: Tensor):
        self.x = x
        # 计算 s = 1 / (1 + exp(-x))
        self.data = 1.0 / (1.0 + np.exp(-x.data))

        # 创建并返回链接好的输出 Tensor
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self):
        # 梯度计算：dL/dx = dL/d(sig) * d(sig)/dx
        # d(sig)/dx = sig(x) * (1 - sig(x))

        # self.data 就是 sig(x)
        # self.grad 就是 dL/d(sig)
        grad_local = self.data * (1.0 - self.data)
        grad_to_pass = self.grad * grad_local

        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = np.zeros_like(self.x.data)
            self.x.grad += grad_to_pass

    def _get_inputs(self):
        return [self.x]

def _im2col(input_data, kernel_h, kernel_w, stride=1, padding=0):
    """
    将 N*C*H*W 的输入数据转换为 Im2Col 矩阵。
    """
    N, C, H, W = input_data.shape#atch Size (N)、通道数 (C)、高度 (H) 和宽度 (W)。
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    #  Padding在 H 和 W 维度上对输入数据进行零填充。
    img = np.pad(input_data, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')

    # 确定 Im2Col 矩阵的索引
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=input_data.dtype)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            # 使用 NumPy 的 stride tricks 来避免循环 (这里为了清晰，保留循环结构)
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    #  Reshape 到 (N*Out_H*Out_W, C*K_H*K_W)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col,  H, W # 返回原始 H, W 用于 col2im


def _col2im(col, input_shape, kernel_h, kernel_w, stride=1, padding=0):
    """
    将 Im2Col 矩阵转换回 N*C*H*W 的梯度形状。
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    # Reshape Col 矩阵
    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)

    # 初始化带 Padding 的图像（用于累加梯度）
    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1), dtype=col.dtype)

    # 散布梯度
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 移除 Padding
    return img[:, :, padding:H + padding, padding:W + padding]


class Conv2dOp(Function):
    """
    负责 2D 卷积的 Op 实现。
    """

    def __init__(self):
        super().__init__()
        self.x = None  # 输入 Tensor X
        self.w = None  # 权重 Tensor W
        self.b = None  # 偏置 Tensor B (可选)
        self.stride = None
        self.padding = None
        self.col = None  # Im2Col 后的输入矩阵（用于反向传播）
        self.x_shape = None  # 输入 X 的形状
        self.data = None  # 输出特征图 O
        self.grad = None  # 上游梯度 dL/dO

    def forward(self, x: Tensor, w: Tensor, b: Tensor = None, stride: int = 1, padding: int = 0):
        #  保存上下文
        self.x = x
        self.w = w
        self.b = b
        self.stride = stride
        self.padding = padding
        self.x_shape = x.shape()

        # 权重 W 的形状: Out_C, In_C, K_H, K_W
        FN, C, KH, KW = w.shape()
        N, C, H, W = x.shape()

        # Im2Col 转换输入
        col, _, _ = _im2col(x.data, KH, KW, stride, padding)
        self.col = col  # 保存用于反向传播

        # 重塑权重 W 为 (Out_C, C*K_H*K_W) 并转置
        W_reshaped = w.data.reshape(FN, -1).T

        #  矩阵乘法: (N*Out_H*Out_W, C*K_H*K_W) @ (C*K_H*K_W, Out_C) -> (N*Out_H*Out_W, Out_C)
        out_flat = np.dot(col, W_reshaped)

        #  添加偏置 B (如果存在)
        if b is not None:
            out_flat += b.data  # 广播机制

        #  Reshape 回 N*Out_C*Out_H*Out_W
        out_h = (H + 2 * padding - KH) // stride + 1
        out_w = (W + 2 * padding - KW) // stride + 1
        self.data = out_flat.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        #  创建输出 Tensor
        requires_grad = x.requires_grad or w.requires_grad or (b and b.requires_grad)
        output_tensor = Tensor(self.data, requires_grad=requires_grad, creator=self)
        return output_tensor

    def backward(self):
        # 梯度 dL/dO
        dL_dO = self.grad

        FN, C, KH, KW = self.w.shape()
        N, _, H, W = self.x_shape

        # 1. 扁平化上游梯度 dL/dO: (N*Out_H*Out_W, Out_C)
        dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 2. 计算对偏置的梯度 dL/dB
        if self.b is not None and self.b.requires_grad:
            dL_dB = np.sum(dL_dO, axis=(0, 2, 3))  # 在 N, H, W 轴求和
            if self.b.grad is None: self.b.grad = np.zeros_like(dL_dB)
            self.b.grad += dL_dB.reshape(self.b.shape())  # 确保形状匹配 (1, Out_C)

        # 3. 计算对权重的梯度 dL/dW = X_col^T @ dL/dO_flat
        if self.w.requires_grad:
            # col.T @ dL_dO_flat -> (C*K_H*K_W, N*Out_H*Out_W) @ (N*Out_H*Out_W, Out_C)
            # 结果形状: (C*K_H*K_W, Out_C)
            dL_dW_flat = np.dot(self.col.T, dL_dO_flat)
            dL_dW = dL_dW_flat.T.reshape(FN, C, KH, KW)
            if self.w.grad is None: self.w.grad = np.zeros_like(dL_dW)
            self.w.grad += dL_dW

        # 4. 计算对输入的梯度 dL/dX = dL/dO_flat @ W_reshaped^T
        if self.x.requires_grad:
            # W_reshaped = (C*K_H*K_W, Out_C)
            W_reshaped = self.w.data.reshape(FN, -1).T

            # dL_dO_flat @ W_reshaped.T -> (N*Out_H*Out_W, Out_C) @ (Out_C, C*K_H*K_W)
            # 结果形状: (N*Out_H*Out_W, C*K_H*K_W) -> 相当于 dL/dCol
            dL_dCol = np.dot(dL_dO_flat, W_reshaped.T)

            # 5. Col2Im 转换回 N*C*H*W
            dL_dX = _col2im(dL_dCol, self.x_shape, KH, KW, self.stride, self.padding)

            if self.x.grad is None: self.x.grad = np.zeros_like(dL_dX)
            self.x.grad += dL_dX

    def _get_inputs(self):
        parents = [self.x, self.w]
        if self.b is not None:
            parents.append(self.b)
        return parents

# =========================================================================
# MaxPoolOp
# =========================================================================

class MaxPoolOp(Function):
    """
    负责 Max Pooling 的 Op 实现。
    """

    def __init__(self):
        super().__init__()
        self.x = None
        self.mask = None  # 关键：保存 Max 元素的索引
        self.x_shape = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.data = None
        self.grad = None

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

        # 1. Im2Col 转换输入
        col, _, _ = _im2col(x.data, KH, KW, stride=self.stride, padding=self.padding)

        # 2. Reshape 为 (N*Out_H*Out_W, C, K*K)
        col_reshaped = col.reshape(-1, C, KH * KW)

        # 3. Max 计算
        self.data = np.max(col_reshaped, axis=2)

        # 4. 记录 Mask (最大值索引)
        # argmax 返回的是 K*K 维度的索引
        argmax_idx = np.argmax(col_reshaped, axis=2)
        self.mask = argmax_idx

        # 5. Reshape 回 N*C*Out_H*Out_W
        self.data = self.data.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # 6. 创建输出 Tensor
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self):
        if self.x.requires_grad:
            # 1. 扁平化上游梯度 dL/dO: (N*Out_H*Out_W, C)
            dL_dO = self.grad
            N, C, Out_H, Out_W = dL_dO.shape
            dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(N * Out_H * Out_W, C)

            KH, KW = self.kernel_size, self.kernel_size

            # 2. 初始化 dL/dCol 矩阵 (与 Col 矩阵形状相同)
            dL_dCol = np.zeros((dL_dO_flat.shape[0], C * KH * KW), dtype=dL_dO.dtype)

            # 3. 散布梯度 (Scatter): 使用 Mask 将 dL/dO 放入 dL/dCol 的对应位置
            # dL_dO_flat 的每个元素都是要散布的值，self.mask 是散布的目标索引

            # 创建用于索引的线性索引
            idx_range = np.arange(dL_dCol.shape[0])[:, None]
            c_range = np.arange(C)[None, :]

            # 使用 Mask 索引来填充 dL/dCol
            # 目标位置的线性索引 = 对应通道起始索引 + Mask 索引
            target_indices = c_range * (KH * KW) + self.mask

            # 散布
            dL_dCol[idx_range, target_indices] = dL_dO_flat

            # 4. Col2Im 转换回 N*C*H*W
            dL_dX = _col2im(dL_dCol, self.x_shape, KH, KW, self.stride, padding=self.padding)

            if self.x.grad is None: self.x.grad = np.zeros_like(dL_dX)
            self.x.grad += dL_dX

    def _get_inputs(self):
        return [self.x]
class MinPoolOp(Function):
    """
    负责 Min Pooling 的 Op 实现。
    (与 MaxPoolOp 逻辑相同，只-是-将 max/argmax 替换为 min/argmin)
    """

    def __init__(self):
        super().__init__()
        self.x = None
        self.mask = None  # 关键：保存 Min 元素的索引
        self.x_shape = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.data = None
        self.grad = None

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

        # 1. Im2Col 转换输入
        col, _, _ = _im2col(x.data, KH, KW, stride=self.stride, padding=self.padding)

        # 2. Reshape 为 (N*Out_H*Out_W, C, K*K)
        col_reshaped = col.reshape(-1, C, KH * KW)

        # 3. Min 计算
        self.data = np.min(col_reshaped, axis=2) # <--- 修正：使用 np.min

        # 4. 记录 Mask (最小值索引)
        argmin_idx = np.argmin(col_reshaped, axis=2) # <--- 修正：使用 np.argmin
        self.mask = argmin_idx

        # 5. Reshape 回 N*C*Out_H*Out_W
        self.data = self.data.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # 6. 创建输出 Tensor
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self):
        if self.x.requires_grad:
            # 1. 扁平化上游梯度 dL/dO: (N*Out_H*Out_W, C)
            dL_dO = self.grad
            N, C, Out_H, Out_W = dL_dO.shape
            dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(N * Out_H * Out_W, C)

            KH, KW = self.kernel_size, self.kernel_size

            # 2. 初始化 dL/dCol 矩阵
            dL_dCol = np.zeros((dL_dO_flat.shape[0], C * KH * KW), dtype=dL_dO.dtype)

            # 3. 散布梯度 (Scatter): 使用 Mask 将 dL/dO 放入 dL/dCol 的对应位置
            idx_range = np.arange(dL_dCol.shape[0])[:, None]
            c_range = np.arange(C)[None, :]

            target_indices = c_range * (KH * KW) + self.mask # <--- 使用 argmin (self.mask)

            dL_dCol[idx_range, target_indices] = dL_dO_flat[:, c_range]

            # 4. Col2Im 转换回 N*C*H*W
            dL_dX = _col2im(dL_dCol, self.x_shape, KH, KW, self.stride, padding=self.padding)

            if self.x.grad is None: self.x.grad = np.zeros_like(dL_dX)
            self.x.grad += dL_dX

    def _get_inputs(self):
        return [self.x]
# =========================================================================
# AvgPoolOp
# =========================================================================

class AvgPoolOp(Function):
    """
    负责 Average Pooling 的 Op 实现。
    """

    def __init__(self):
        super().__init__()
        self.x = None
        self.x_shape = None
        self.pool_area = None  # 窗口大小 K*K
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.data = None
        self.grad = None

    def forward(self, x: Tensor, kernel_size, stride, padding=0):
        self.x = x
        self.x_shape = x.shape()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        KH, KW = kernel_size, kernel_size
        self.pool_area = KH * KW

        # 1. Im2Col 转换输入
        col, _, _ = _im2col(x.data, KH, KW, stride=stride, padding=self.padding)

        # 2. 平均计算
        self.data = np.mean(col.reshape(-1, x.shape()[1], self.pool_area), axis=2)

        # 3. Reshape 回 N*C*Out_H*Out_W
        N, C, H, W = x.shape()
        out_h = (H + 2 * self.padding - KH) // stride + 1
        out_w = (W + 2 * self.padding - KW) // stride + 1
        self.data = self.data.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # 4. 创建输出 Tensor
        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self):
        if self.x.requires_grad:
            # 1. 扁平化上游梯度 dL/dO: (N*Out_H*Out_W, C)
            dL_dO = self.grad
            N, C, Out_H, Out_W = dL_dO.shape
            dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(N * Out_H * Out_W, C)

            KH, KW = self.kernel_size, self.kernel_size

            # 2. 梯度分配：将 dL/dO 的值均匀分配到窗口中的所有输入元素
            # dL/dX 的每个窗口元素都接收 dL/dO / K^2
            dL_dO_distributed = dL_dO_flat / self.pool_area

            # 3. 复制梯度到 dL/dCol
            dL_dCol = np.repeat(dL_dO_distributed, self.pool_area, axis=1)

            # 4. Col2Im 转换回 N*C*H*W
            dL_dX = _col2im(dL_dCol, self.x_shape, KH, KW, self.stride, padding=self.padding)

            if self.x.grad is None: self.x.grad = np.zeros_like(dL_dX)
            self.x.grad += dL_dX

    def _get_inputs(self):
        return [self.x]


class ReshapeOp(Function):
    def __init__(self):
        super().__init__()
        self.x = None
        self.x_shape = None

    def forward(self, x: Tensor, *shape):
        self.x = x
        self.x_shape = x.shape()  # 保存原始形状
        self.data = x.data.reshape(*shape)

        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self):
        if self.x.requires_grad:
            # 将梯度 reshape 回原始形状
            grad_to_pass = self.grad.reshape(self.x_shape)
            if self.x.grad is None:
                self.x.grad = np.zeros_like(grad_to_pass)
            self.x.grad += grad_to_pass

    def _get_inputs(self):
        return [self.x]


class MSE(Function):
    def __init__(self):
        super().__init__()
        self.y_prediction = None
        self.y_true = None
        self.data = None
        self.grad = None

    def forward(self, y_prediction: Tensor, y_true: Tensor):
        self.y_prediction = y_prediction
        self.y_true = y_true

        # 计算均方误差
        # (N,)
        error = self.y_prediction.data - self.y_true.data
        # (N,)
        squared_error = error ** 2
        # scalar
        self.data = np.mean(squared_error)

        # 损失总是需要梯度（相对于 y_prediction）
        requires_grad = y_prediction.requires_grad
        output_tensor = Tensor(self.data, requires_grad=requires_grad, creator=self)
        return output_tensor

    def backward(self):
        # MSE = (1/N) * sum((y_pred - y_true)^2)
        # d(MSE)/d(y_pred) = (1/N) * 2 * (y_pred - y_true)

        # 确保 y_true 形状与 y_prediction 匹配
        y_true_data = self.y_true.data.reshape(self.y_prediction.data.shape)

        # self.grad 是从 loss.backward() 传来的，标量 1.0
        # grad_local 是 MSE 对 y_prediction 的局部梯度
        N = self.y_prediction.data.size
        grad_local = (2.0 / N) * (self.y_prediction.data - y_true_data)

        # 链式法则
        grad_to_pass = self.grad * grad_local

        if self.y_prediction.requires_grad:
            if self.y_prediction.grad is None:
                self.y_prediction.grad = np.zeros_like(self.y_prediction.data)
            self.y_prediction.grad += grad_to_pass

    def _get_inputs(self):
        return [self.y_prediction, self.y_true]


# 文件: function.py
# ... (在 MSE 类的定义之后) ...

class LogSoftmaxOp(Function):
    """
    负责 Log(Softmax(x)) 的 Op 实现。
    这是一个数值稳定的实现。
    """

    def __init__(self):
        super().__init__()
        self.x = None
        # self.data 将存储 log_softmax 的输出

    def forward(self, x: Tensor):
        # x (logits) 形状: (N, C)
        self.x = x

        # 数值稳定技巧：
        # log(sum(exp(x_i))) = log(sum(exp(x_i - max_x))) + max_x
        # log_softmax(x_i) = x_i - log(sum(exp(x_j)))
        # log_softmax(x_i) = (x_i - max_x) - log(sum(exp(x_j - max_x)))

        max_x = self.x.data.max(axis=1, keepdims=True)
        stable_x = self.x.data - max_x
        exp_stable_x = np.exp(stable_x)
        sum_exp = exp_stable_x.sum(axis=1, keepdims=True)
        log_sum_exp = np.log(sum_exp)

        self.data = stable_x - log_sum_exp  # 这就是 log_softmax 的结果

        output_tensor = Tensor(self.data, requires_grad=x.requires_grad, creator=self)
        return output_tensor

    def backward(self):
        # dL/dx = dL/d_log_softmax - (sum(dL/d_log_softmax) * softmax(x))
        if self.x.requires_grad:
            # self.data 是 log_softmax, exp(self.data) 就是 softmax
            softmax = np.exp(self.data)

            # self.grad 是 dL/d_log_softmax
            sum_grad = self.grad.sum(axis=1, keepdims=True)

            grad_to_pass = self.grad - (softmax * sum_grad)

            if self.x.grad is None:
                self.x.grad = np.zeros_like(self.x.data)
            self.x.grad += grad_to_pass

    def _get_inputs(self):
        return [self.x]


class NLLLossOp(Function):
    """
    负对数似然损失 (NLL Loss) 的 Op 实现。
    它期望输入为 LogSoftmax 概率和目标类别索引。
    """

    def __init__(self):
        super().__init__()
        self.log_probs = None  # 输入 (N, C)
        self.target = None  # 目标 (N,)
        self.N = None
        self.C = None
        # self.data 将存储标量损失

    def forward(self, log_probs: Tensor, target: Tensor):
        # log_probs 形状 (N, C)
        # target 形状 (N,)，包含 0 到 C-1 的索引
        self.log_probs = log_probs
        self.target = target
        self.N, self.C = self.log_probs.shape()

        # "挑选" 出对应 target 索引的 log 概率
        # log_probs.data[ (0, 1, 2, ... N-1), (t_0, t_1, t_2, ... t_N-1) ]
        picked_log_probs = self.log_probs.data[np.arange(self.N), self.target.data]

        # L = -mean(picked_log_probs)
        self.data = -np.mean(picked_log_probs)

        # 损失总是需要梯度（相对于 log_probs）
        output_tensor = Tensor(self.data, requires_grad=self.log_probs.requires_grad, creator=self)
        return output_tensor

    def backward(self):
        # L = (1/N) * sum(-log_probs[i, target[i]])
        # dL/d_log_probs[i, j] = 0 (如果 j != target[i])
        # dL/d_log_probs[i, j] = -(1/N) (如果 j == target[i])

        if self.log_probs.requires_grad:
            # self.grad 是标量 1.0
            grad_to_pass = np.zeros_like(self.log_probs.data)

            # 在 target 索引处放置梯度
            grad_to_pass[np.arange(self.N), self.target.data] = -1.0 / self.N

            # 链式法则
            grad_to_pass *= self.grad

            if self.log_probs.grad is None:
                self.log_probs.grad = np.zeros_like(self.log_probs.data)
            self.log_probs.grad += grad_to_pass

            # target (标签) 不需要梯度

    def _get_inputs(self):
        return [self.log_probs, self.target]