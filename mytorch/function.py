import time
import numpy as np
from .tensor import Tensor

# 尝试导入 CuPy 以支持 GPU 加速
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
from .jit_ir import TracerState, IRNode

# =========================================================================
# 1. 辅助工具函数 (Utils)
# =========================================================================

def _to_2tuple(value):
    """
    将标量或元组统一转换为 2D 元组。
    例如: 3 -> (3, 3), (1, 2) -> (1, 2)
    """
    if isinstance(value, tuple):
        return value
    return value, value


def _im2col(input_data, kernel_h, kernel_w, stride, padding, xp=np):
    """
    极速版 im2col：针对标准对称卷积优化。
    去掉 dilation 逻辑以减少 Python 层的循环开销，提升基础卷积性能。
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    # 在高度和宽度维度进行填充
    img = xp.pad(input_data, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
    col = xp.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=input_data.dtype)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col, H, W


def _col2im(col, input_shape, kernel_h, kernel_w, stride, padding, xp=np):
    """
    极速版 col2im：将矩阵转换回图像形状的梯度。
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


def _im2col_ext(input_data, kernel_h, kernel_w, stride=1, padding=0, dilation=1, xp=np):
    """
    扩展版 im2col：完整支持非对称 stride/padding 以及 dilation。
    用于转置卷积或空洞卷积。
    """
    N, C, H, W = input_data.shape
    stride_h, stride_w = _to_2tuple(stride)
    pad_h, pad_w = _to_2tuple(padding)
    dil_h, dil_w = _to_2tuple(dilation)

    eff_kh = dil_h * (kernel_h - 1) + 1
    eff_kw = dil_w * (kernel_w - 1) + 1
    out_h = (H + 2 * pad_h - eff_kh) // stride_h + 1
    out_w = (W + 2 * pad_w - eff_kw) // stride_w + 1

    img = xp.pad(input_data, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], 'constant')
    col = xp.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=input_data.dtype)

    for y in range(kernel_h):
        y_base = y * dil_h
        y_max = y_base + stride_h * out_h
        for x in range(kernel_w):
            x_base = x * dil_w
            x_max = x_base + stride_w * out_w
            col[:, :, y, x, :, :] = img[:, :, y_base:y_max:stride_h, x_base:x_max:stride_w]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col, H, W, out_h, out_w


def _col2im_ext(col, input_shape, kernel_h, kernel_w, stride=1, padding=0, dilation=1, xp=np):
    """
    扩展版 col2im：支持 dilation 的梯度还原。
    """
    N, C, H, W = input_shape
    stride_h, stride_w = _to_2tuple(stride)
    pad_h, pad_w = _to_2tuple(padding)
    dil_h, dil_w = _to_2tuple(dilation)

    eff_kh = dil_h * (kernel_h - 1) + 1
    eff_kw = dil_w * (kernel_w - 1) + 1
    out_h = (H + 2 * pad_h - eff_kh) // stride_h + 1
    out_w = (W + 2 * pad_w - eff_kw) // stride_w + 1

    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    img = xp.zeros((N, C, H + 2 * pad_h + stride_h - 1, W + 2 * pad_w + stride_w - 1), dtype=col.dtype)

    for y in range(kernel_h):
        y_base = y * dil_h
        y_max = y_base + stride_h * out_h
        for x in range(kernel_w):
            x_base = x * dil_w
            x_max = x_base + stride_w * out_w
            img[:, :, y_base:y_max:stride_h, x_base:x_max:stride_w] += col[:, :, y, x, :, :]

    return img[:, :, pad_h:H + pad_h, pad_w:W + pad_w]

# =========================================================================
# GEMM 工具函数：统一替换卷积中的 einsum
# =========================================================================

def _contiguous(x, xp):
    """
    保证传入 matmul 的矩阵是连续内存。
    对 CPU 是 np.ascontiguousarray；
    对 GPU 是 cp.ascontiguousarray。
    """
    if hasattr(xp, "ascontiguousarray"):
        return xp.ascontiguousarray(x)
    return np.ascontiguousarray(x)


def _gemm(A, B, xp):
    """
    统一 GEMM 入口。
    这里实际调用 np.matmul / cp.matmul。
    """
    A = _contiguous(A, xp)
    B = _contiguous(B, xp)
    return xp.matmul(A, B)


def _grouped_gemm_forward(col_grouped, W_grouped, xp):
    """
    分组 GEMM forward。

    col_grouped: [M, G, K]
    W_grouped:   [G, F, K]

    输出:
    out_grouped: [M, G, F]

    对每个 group 执行:
        out[:, g, :] = col[:, g, :] @ W[g].T
    """
    M, G, K = col_grouped.shape
    G2, F, K2 = W_grouped.shape

    if G != G2 or K != K2:
        raise ValueError(
            f"grouped GEMM shape 不匹配: "
            f"col={col_grouped.shape}, W={W_grouped.shape}"
        )

    out_grouped = xp.empty((M, G, F), dtype=col_grouped.dtype)

    for g in range(G):
        out_grouped[:, g, :] = _gemm(
            col_grouped[:, g, :],
            W_grouped[g].T,
            xp
        )

    return out_grouped


def _grouped_gemm_backward(col_grouped, dO_grouped, W_grouped, xp):
    """
    分组 GEMM backward。

    forward:
        out[:, g, :] = col[:, g, :] @ W[g].T

    backward:
        dW[g]   = dO[:, g, :].T @ col[:, g, :]
        dCol[g] = dO[:, g, :]   @ W[g]

    col_grouped: [M, G, K]
    dO_grouped:  [M, G, F]
    W_grouped:   [G, F, K]

    返回:
        dW_grouped:   [G, F, K]
        dCol_grouped: [M, G, K]
    """
    M, G, K = col_grouped.shape
    M2, G2, F = dO_grouped.shape
    G3, F2, K2 = W_grouped.shape

    if M != M2 or G != G2 or G != G3 or F != F2 or K != K2:
        raise ValueError(
            f"grouped GEMM backward shape 不匹配: "
            f"col={col_grouped.shape}, dO={dO_grouped.shape}, W={W_grouped.shape}"
        )

    dW_grouped, dCol_grouped = _grouped_gemm_backward_select(
        col_grouped,
        dO_grouped,
        W_grouped,
        xp,
        need_w=True,
        need_col=True,
    )

    return dW_grouped, dCol_grouped


def _grouped_gemm_backward_select(
    col_grouped,
    dO_grouped,
    W_grouped,
    xp,
    need_w=True,
    need_col=True,
):
    """
    Optional-output grouped GEMM backward.

    This avoids allocating the large dCol buffer when input gradients are not
    needed, and avoids allocating dW when weights do not require gradients.
    """
    M, G, K = col_grouped.shape
    M2, G2, F = dO_grouped.shape
    G3, F2, K2 = W_grouped.shape

    if M != M2 or G != G2 or G != G3 or F != F2 or K != K2:
        raise ValueError(
            f"grouped GEMM backward shape 不匹配: "
            f"col={col_grouped.shape}, dO={dO_grouped.shape}, W={W_grouped.shape}"
        )

    dW_grouped = xp.empty((G, F, K), dtype=W_grouped.dtype) if need_w else None
    dCol_grouped = xp.empty((M, G, K), dtype=col_grouped.dtype) if need_col else None

    for g in range(G):
        if need_w:
            dW_grouped[g] = _gemm(
                dO_grouped[:, g, :].T,
                col_grouped[:, g, :],
                xp
            )

        if need_col:
            dCol_grouped[:, g, :] = _gemm(
                dO_grouped[:, g, :],
                W_grouped[g],
                xp
            )

    return dW_grouped, dCol_grouped


def _conv2d_gemm_forward(col, w_data, groups, FN, C_group, KH, KW, xp):
    """
    Conv2d forward 的统一 GEMM 入口。

    groups == 1:
        col:  [M, K]
        W:    [FN, K]
        out:  [M, FN]

    groups > 1:
        col:  [M, G, K]
        W:    [G, F, K]
        out:  [M, FN]
    """
    K = C_group * KH * KW

    if groups == 1:
        col_2d = _contiguous(col.reshape(-1, K), xp)
        W_2d = _contiguous(w_data.reshape(FN, K), xp)

        out_flat = _gemm(col_2d, W_2d.T, xp)
        return out_flat, col_2d

    col_grouped = _contiguous(col.reshape(-1, groups, K), xp)
    W_grouped = _contiguous(
        w_data.reshape(groups, FN // groups, K),
        xp
    )

    out_grouped = _grouped_gemm_forward(
        col_grouped,
        W_grouped,
        xp
    )

    out_flat = out_grouped.reshape(-1, FN)

    return out_flat, col_grouped


def _conv2d_gemm_backward(col, dO_flat, w_data, groups, FN, C_group, KH, KW, xp):
    """
    Conv2d backward 的统一 GEMM 入口。

    返回:
        dW:   shape 与 w_data 相同
        dCol: shape = [M, C_group * KH * KW * groups]
    """
    return _conv2d_gemm_backward_select(
        col,
        dO_flat,
        w_data,
        groups,
        FN,
        C_group,
        KH,
        KW,
        xp,
        need_w=True,
        need_col=True,
    )


def _conv2d_gemm_backward_select(
    col,
    dO_flat,
    w_data,
    groups,
    FN,
    C_group,
    KH,
    KW,
    xp,
    need_w=True,
    need_col=True,
):
    """
    Optional-output Conv2d GEMM backward.

    Returns:
        dW or None
        dCol or None
    """
    if not need_w and not need_col:
        return None, None

    K = C_group * KH * KW
    dO_flat = _contiguous(dO_flat, xp)

    if groups == 1:
        col_2d = _contiguous(col.reshape(-1, K), xp) if need_w else None
        W_2d = _contiguous(w_data.reshape(FN, K), xp) if need_col else None

        dW = _gemm(dO_flat.T, col_2d, xp).reshape(FN, C_group, KH, KW) if need_w else None
        dCol = _gemm(dO_flat, W_2d, xp) if need_col else None

        return dW, dCol

    out_channels_per_group = FN // groups

    col_grouped = _contiguous(col.reshape(-1, groups, K), xp)
    dO_grouped = _contiguous(
        dO_flat.reshape(-1, groups, out_channels_per_group),
        xp
    )

    W_grouped = _contiguous(
        w_data.reshape(groups, out_channels_per_group, K),
        xp
    )

    dW_grouped, dCol_grouped = _grouped_gemm_backward_select(
        col_grouped,
        dO_grouped,
        W_grouped,
        xp,
        need_w=need_w,
        need_col=need_col,
    )

    dW = dW_grouped.reshape(FN, C_group, KH, KW) if need_w else None
    dCol = dCol_grouped.reshape(-1, groups * K) if need_col else None

    return dW, dCol
# =========================================================================
# 2. Function 基类
# =========================================================================

class Function:
    """
    自动求导操作的抽象基类。
    增加了全局设备一致性检查拦截（覆盖 args 和 kwargs）。
    """
    def __init__(self):
        self.data = None
        self.grad = None
        self.xp = np

    def __call__(self, *args, **kwargs):
 
        """
        拦截器：在进入 forward 之前，确保所有输入 Tensor 都在同一设备上。
        """
        base_xp = None
        # 遍历位置参数和关键字参数
        all_args = list(args) + list(kwargs.values())
        
        for arg in all_args:
            if isinstance(arg, Tensor):
                if base_xp is None:
                    base_xp = arg.xp
                    self.xp = base_xp
                elif arg.xp is not base_xp:
                    raise RuntimeError(
                        "RuntimeError: 期望所有 Tensor 都在同一设备上 (CPU 或 GPU)，"
                        "但发现了混用的情况，请先调用 .cuda() 或 .cpu() 统一设备。"
                    )
        # 2. 策略 B 核心：无论如何，先执行真实的 Eager 前向计算！
        result = self.forward(*args, **kwargs)

        # 3. JIT 拦截记录阶段 (窃取真实运算结果的 shape)
        if TracerState._is_tracing and isinstance(result, Tensor):
            op_name = self.__class__.__name__
            
            # 收集输入节点
            ir_inputs = []
            for arg in args:
                # 如果输入是 Tensor 且带了 ir_node 标记，说明它是上游传下来的
                if isinstance(arg, Tensor) and hasattr(arg, 'ir_node') and arg.ir_node is not None:
                    ir_inputs.append(arg.ir_node) 
                else:
                    ir_inputs.append(arg) # 记录普通标量（如 stride=1）
                    
            # 处理 kwargs
            for k, v in kwargs.items():
                ir_inputs.append(v)
            
            # 创建新节点，利用 result 窃取真实的 shape 和 dtype
            node = IRNode(
                op_name=op_name, 
                inputs=ir_inputs, 
                output_shape=result.shape(), 
                output_dtype=str(result.dtype()),
                origin_func=self
            )
            TracerState._current_graph.add_node(node)
            
            # 将新节点挂载到输出 Tensor 上，传递给下一个算子
            result.ir_node = node 

        return result

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
        raise NotImplementedError

    def _get_inputs(self):
        raise NotImplementedError
    
    def infer_shape(self, *args, **kwargs):
        """需要为不同算子重写形状推导逻辑"""
        # 默认行为：如果是 Add/ReLU 等逐元素操作，输出 shape 等于输入 shape
        if hasattr(args[0], 'shape'):
            return args[0].shape() if callable(args[0].shape) else args[0].shape
        return None
# =========================================================================
# 3. 基础数学算子
# =========================================================================

class Add(Function):
    def forward(self, a, b):
        self.a, self.b = a, b
        self.data = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad
        return Tensor(self.data, requires_grad=requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
        
        # 针对 a 和 b 分别处理梯度广播还原
        for tensor in [self.a, self.b]:
            if tensor.requires_grad:
                g = self.grad
                # 1. 维度数不同，先在前面求和以对齐维度
                while g.ndim > tensor.data.ndim:
                    g = g.sum(axis=0)
                # 2. 维度数相同但大小为 1 的轴，进行求和还原
                for i, dim in enumerate(tensor.data.shape):
                    if dim == 1 and g.shape[i] > 1:
                        g = g.sum(axis=i, keepdims=True)
                
                if tensor.grad is None:
                    tensor.grad = self.xp.zeros_like(tensor.data)
                tensor.grad += g

    def _get_inputs(self):
        return [self.a, self.b]

class MatMul(Function):
    """
    矩阵乘法算子 (对应于 np.matmul 或 @ 运算符)
    主要用于 Linear (全连接层) 的前向与反向传播计算。
    """
    def forward(self, a, b):
        self.a, self.b = a, b
        self.data = self.xp.matmul(a.data, b.data)
        requires_grad = a.requires_grad or b.requires_grad
        return Tensor(self.data, requires_grad=requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
            
        # 1. 计算对 a 的梯度: grad @ b.T
        if self.a.requires_grad:
            if self.a.grad is None:
                self.a.grad = self.xp.zeros_like(self.a.data)
            self.a.grad += self.xp.matmul(self.grad, self.b.data.swapaxes(-1, -2))
            
        # 2. 计算对 b 的梯度: a.T @ grad
        if self.b.requires_grad:
            if self.b.grad is None:
                self.b.grad = self.xp.zeros_like(self.b.data)
                
            grad_b = self.xp.matmul(self.a.data.swapaxes(-1, -2), self.grad)
            # 处理 Batch 维度（例如 a 是多维张量时，需要将前方多出的维度求和掉）
            while grad_b.ndim > self.b.data.ndim:
                grad_b = self.xp.sum(grad_b, axis=0)
                
            self.b.grad += grad_b

    def _get_inputs(self):
        return [self.a, self.b]
    
class MSE(Function):
    def forward(self, y_pred: Tensor, y_true: Tensor):
        self.y_pred = y_pred
        self.y_true = y_true
        
        # 使用底层数据计算 MSE
        diff = y_pred.data - y_true.data
        self.data = self.xp.mean(diff ** 2)
        
        # 修正 1：只要预测值或真实值任一需要梯度，输出张量就需要梯度
        requires_grad = y_pred.requires_grad or y_true.requires_grad
        return Tensor(self.data, requires_grad=requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
            
        # 修正 2：提取公共计算部分
        N = self.y_pred.data.size
        # grad_local 是 Loss 对 (y_pred - y_true) 的局部导数
        grad_local = (2.0 / N) * (self.y_pred.data - self.y_true.data)
            
        # 对预测值计算梯度并累加
        if self.y_pred.requires_grad:
            if self.y_pred.grad is None:
                self.y_pred.grad = self.xp.zeros_like(self.y_pred.data)
            # 链式法则：上游梯度 * 局部梯度
            self.y_pred.grad += self.grad * grad_local

        # 修正 3：支持对 y_true 计算梯度
        if self.y_true.requires_grad:
            if self.y_true.grad is None:
                self.y_true.grad = self.xp.zeros_like(self.y_true.data)
            # 对 y_true 的导数是 y_pred 的相反数 (因为有 -y_true)
            self.y_true.grad -= self.grad * grad_local

    def _get_inputs(self):
        return [self.y_pred, self.y_true]


class Sum(Function):
    """
    全量求和算子，对应 Tensor.sum()。
    """
    def forward(self, x):
        self.x = x
        self.data = self.xp.sum(x.data)
        return Tensor(self.data, requires_grad=x.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
        
        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            # 梯度的反向传播是全量广播
            self.x.grad += self.xp.full(self.x.data.shape, self.grad)

    def _get_inputs(self):
        return [self.x]


class ReshapeOp(Function):
    def forward(self, x, *shape):
        self.x = x
        self.old_shape = x.shape()
        self.data = x.data.reshape(*shape)
        return Tensor(self.data, requires_grad=x.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
        
        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += self.grad.reshape(self.old_shape)

    def _get_inputs(self):
        return [self.x]


# =========================================================================
# 4. 激活函数算子
# =========================================================================

class ReLU(Function):
    def forward(self, x):
        self.x = x
        self.data = self.xp.maximum(0, x.data)
        return Tensor(self.data, requires_grad=x.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
        
        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            # x > 0 处导数为 1，否则为 0
            self.x.grad += self.grad * (self.x.data > 0)

    def _get_inputs(self):
        return [self.x]


class ELU(Function):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        mask = x.data > 0
        self.data = self.xp.where(
            mask, 
            x.data, 
            self.alpha * (self.xp.exp(x.data) - 1)
        )
        return Tensor(self.data, requires_grad=x.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
        
        if self.x.requires_grad:
            mask = self.x.data > 0
            # 局部导数：x>0 为 1，x<=0 为 alpha * exp(x)
            grad_local = self.xp.where(mask, 1.0, self.alpha * self.xp.exp(self.x.data))
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += self.grad * grad_local

    def _get_inputs(self):
        return [self.x]


class Sigmoid(Function):
    def forward(self, x):
        self.x = x
        self.data = 1.0 / (1.0 + self.xp.exp(-x.data))
        return Tensor(self.data, requires_grad=x.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
        
        if self.x.requires_grad:
            # 导数 = s * (1 - s)
            grad_local = self.data * (1.0 - self.data)
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += self.grad * grad_local

    def _get_inputs(self):
        return [self.x]

# =========================================================================
# 5. 核心卷积与转置卷积 (Conv2d, ConvTranspose2d)
# =========================================================================

class Conv2dOp(Function):
    """
    全功能版卷积算子 (终极稳妥版)：
    - 双车道智能路由：对称普通卷积走极速版，非对称/空洞卷积走扩展版。
    - 完整支持 groups (分组卷积/深度可分离卷积)。
    - 彻底适配 GPU (self.xp 动态后端)。
    - 严谨的通道与形状安全检查。
    """
    def __init__(self):
        super().__init__()
        self.x = None
        self.w = None
        self.b = None
        self.stride = None
        self.padding = None
        self.dilation = None
        self.groups = None
        self.col = None
        self.x_shape = None
        self.out_h = None
        self.out_w = None

    def forward(self, x: Tensor, w: Tensor, b: Tensor = None,
                stride=1, padding=0, dilation=1, groups=1):
        self.x = x
        self.w = w
        self.b = b

        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups
        self.x_shape = x.shape()

        FN, C_group, KH, KW = w.shape()
        N, C_total, H, W_in = self.x_shape

        if groups <= 0 or C_total % groups != 0 or FN % groups != 0:
            raise ValueError("Groups 参数设置非法或输入输出通道数无法被整除")

        if C_group != C_total // groups:
            raise ValueError(
                f"权重输入通道与 groups 不匹配："
                f"期望 {C_total // groups}，实际得到 {C_group}"
            )

        use_fast_path = (
            self.dilation == (1, 1)
            and self.stride[0] == self.stride[1]
            and self.padding[0] == self.padding[1]
        )

        if use_fast_path:
            col, _, _ = _im2col(
                x.data,
                KH,
                KW,
                stride=self.stride[0],
                padding=self.padding[0],
                xp=self.xp
            )
            out_h = (H + 2 * self.padding[0] - KH) // self.stride[0] + 1
            out_w = (W_in + 2 * self.padding[0] - KW) // self.stride[0] + 1
        else:
            col, _, _, out_h, out_w = _im2col_ext(
                x.data,
                KH,
                KW,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                xp=self.xp
            )

        # ==========================================================
        # GEMM fast path:
        #   groups == 1 时，直接使用二维矩阵乘法：
        #       col_2d: [N * OH * OW, C * KH * KW]
        #       W_2d:   [FN, C * KH * KW]
        #       out:    [N * OH * OW, FN]
        #
        # ResNet18 的普通卷积基本都是 groups == 1，
        # 所以主干卷积都会走这个路径。
        # ==========================================================
        out_flat, self.col = _conv2d_gemm_forward(
            col=col,
            w_data=w.data,
            groups=groups,
            FN=FN,
            C_group=C_group,
            KH=KH,
            KW=KW,
            xp=self.xp
        )

        if b is not None:
            if b.data.size != FN:
                raise ValueError(
                    f"Bias 元素数量必须等于 out_channels={FN}，"
                    f"实际传入了 {b.data.size}。"
                )
            out_flat += b.data.reshape(1, FN)

        self.data = out_flat.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        self.out_h = out_h
        self.out_w = out_w

        requires_grad = (
            x.requires_grad
            or w.requires_grad
            or (b is not None and b.requires_grad)
        )

        return Tensor(
            self.data,
            requires_grad=requires_grad,
            creator=self
        )
    
    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        dL_dO = self.grad

        FN, C_group, KH, KW = self.w.shape()
        groups = self.groups

        dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 1. Bias 梯度
        if self.b is not None and self.b.requires_grad:
            dL_dB = self.xp.sum(dL_dO_flat, axis=0, keepdims=True)

            if self.b.grad is None:
                self.b.grad = self.xp.zeros_like(self.b.data)

            self.b.grad += dL_dB.reshape(self.b.shape())

        need_w_grad = self.w.requires_grad
        need_x_grad = self.x.requires_grad
        dL_dW = dL_dCol = None

        if need_w_grad or need_x_grad:
            dL_dW, dL_dCol = _conv2d_gemm_backward_select(
                col=self.col,
                dO_flat=dL_dO_flat,
                w_data=self.w.data,
                groups=groups,
                FN=FN,
                C_group=C_group,
                KH=KH,
                KW=KW,
                xp=self.xp,
                need_w=need_w_grad,
                need_col=need_x_grad,
            )

        # 2. Weight 梯度
        if need_w_grad:
            if self.w.grad is None:
                self.w.grad = self.xp.zeros_like(self.w.data)

            self.w.grad += dL_dW

        # 3. Input 梯度
        if need_x_grad:
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)

            use_fast_path = (
                self.dilation == (1, 1)
                and self.stride[0] == self.stride[1]
                and self.padding[0] == self.padding[1]
            )

            if use_fast_path:
                self.x.grad += _col2im(
                    dL_dCol,
                    self.x_shape,
                    KH,
                    KW,
                    stride=self.stride[0],
                    padding=self.padding[0],
                    xp=self.xp
                )
            else:
                self.x.grad += _col2im_ext(
                    dL_dCol,
                    self.x_shape,
                    KH,
                    KW,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    xp=self.xp
                )

    def _get_inputs(self):
        parents = [self.x, self.w]
        if self.b is not None:
            parents.append(self.b)
        return parents


class ConvTranspose2dOp(Function):
    """转置卷积 (反卷积) 算子"""
    def __init__(self):
        super().__init__()
        self.x = None
        self.w = None
        self.b = None
        self.stride = None
        self.padding = None
        self.output_padding = None
        self.dilation = None
        self.groups = None
        self.x_shape = None
        self.x_up_shape = None
        self.col = None
        self.conv_padding = None

    def forward(self, x, w, b=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
        self.x, self.w, self.b = x, w, b
        self.stride, self.padding, self.out_pad, self.dilation, self.groups = _to_2tuple(stride), _to_2tuple(padding), _to_2tuple(output_padding), _to_2tuple(dilation), groups
        N, C_in, H, W = x.shape()
        C_in_w, C_out_g, KH, KW = w.shape()
        
        # ==========================================
        # 参数合法性与形状防御检查
        # ==========================================
        if groups <= 0:
            raise ValueError("groups 必须大于 0")
        if C_in_w != C_in:
            raise ValueError(f"转置卷积权重输入通道数 ({C_in_w}) 与输入 ({C_in}) 不匹配")
        if C_in % groups != 0:
            raise ValueError("in_channels 必须能被 groups 整除")
        if self.out_pad[0] >= self.stride[0] or self.out_pad[1] >= self.stride[1]:
            raise ValueError("output_padding 必须严格小于 stride")
            
        if b is not None and b.data.size != C_out_g * groups:
            raise ValueError(f"bias 数量错误：应为 {C_out_g * groups}，实际为 {b.data.size}")

        # 内部参数计算
        conv_pad_h = self.dilation[0] * (KH - 1) - self.padding[0]
        conv_pad_w = self.dilation[1] * (KW - 1) - self.padding[1]
        
        if conv_pad_h < 0 or conv_pad_w < 0:
            raise ValueError("当前实现要求 padding <= dilation * (kernel_size - 1)")
            
        self.conv_padding = (conv_pad_h, conv_pad_w)

        # 插空放大
        h_up, w_up = (H - 1) * self.stride[0] + 1, (W - 1) * self.stride[1] + 1
        x_up = self.xp.zeros((N, C_in, h_up + self.out_pad[0], w_up + self.out_pad[1]), dtype=x.data.dtype)
        x_up[:, :, ::self.stride[0], ::self.stride[1]] = x.data
        self.x_up_shape = x_up.shape
        
        # 核心逻辑
        col, _, _, out_h, out_w = _im2col_ext(x_up, KH, KW, 1, self.conv_padding, self.dilation, xp=self.xp)
        self.col = col.reshape(-1, groups, (C_in // groups) * KH * KW)
        w_conv = w.data.reshape(groups, C_in // groups, C_out_g, KH, KW).transpose(0, 2, 1, 3, 4)[:, :, :, ::-1, ::-1].reshape(groups, C_out_g, -1)

        self.w_conv = w_conv

        out_grouped = _grouped_gemm_forward(
            self.col,
            self.w_conv,
            self.xp
        )

        out_flat = out_grouped.reshape(-1, C_out_g * groups)        
        if b is not None: 
            out_flat += b.data.reshape(1, -1)
        
        self.data = out_flat.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        # 修正：将 bias 加入 requires_grad 的判定链条中
        req_grad = x.requires_grad or w.requires_grad or (b is not None and b.requires_grad)
        return Tensor(self.data, requires_grad=req_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        _, C_out_g, KH, KW = self.w.shape()

        # grad: [N, C_out_total, OH, OW]
        # dL_dO: [N * OH * OW, groups, C_out_per_group]
        dL_dO = self.grad.transpose(0, 2, 3, 1).reshape(
            -1,
            self.groups,
            C_out_g
        )

        # ==========================================================
        # 1. bias 梯度
        # ==========================================================
        if self.b is not None and self.b.requires_grad:
            if self.b.grad is None:
                self.b.grad = self.xp.zeros_like(self.b.data)

            dL_dB = (
                self.grad
                .transpose(0, 2, 3, 1)
                .reshape(-1, self.grad.shape[1])
                .sum(axis=0)
                .reshape(self.b.shape())
            )

            self.b.grad += dL_dB

        # ==========================================================
        # 2. 统一使用 grouped GEMM backward
        #
        # forward:
        #   out_grouped = col @ w_conv.T
        #
        # backward:
        #   dW_grouped   = dO.T @ col
        #   dCol_grouped = dO @ w_conv
        #
        # 注意：
        #   这里不能放进 if self.w.requires_grad 里面。
        #   因为即使 w 不需要梯度，x 也可能需要 dCol。
        # ==========================================================
        dW_grouped, dCol_grouped = _grouped_gemm_backward(
            self.col,
            dL_dO,
            self.w_conv,
            self.xp
        )

        # ==========================================================
        # 3. weight 梯度
        # ==========================================================
        if self.w.requires_grad:
            # dW_grouped: [groups, C_out_g, C_in_g * KH * KW]
            dL_dW_c = dW_grouped.reshape(
                self.groups,
                C_out_g,
                -1,
                KH,
                KW
            )

            # forward 中 w_conv 的构造是：
            # w.data
            #   -> reshape(groups, C_in_g, C_out_g, KH, KW)
            #   -> transpose(0, 2, 1, 3, 4)
            #   -> kernel 空间翻转
            #
            # backward 这里反向还原。
            dL_dW = (
                dL_dW_c[:, :, :, ::-1, ::-1]
                .transpose(0, 2, 1, 3, 4)
                .reshape(self.w.data.shape)
            )

            if self.w.grad is None:
                self.w.grad = self.xp.zeros_like(self.w.data)

            self.w.grad += dL_dW

        # ==========================================================
        # 4. input 梯度
        # ==========================================================
        if self.x.requires_grad:
            dL_dCol = dCol_grouped.reshape(dL_dO.shape[0], -1)

            dL_dX_up = _col2im_ext(
                dL_dCol,
                self.x_up_shape,
                KH,
                KW,
                1,
                self.conv_padding,
                self.dilation,
                xp=self.xp
            )

            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)

            # 转置卷积 forward 里对输入做过插空：
            # x_up[:, :, ::stride_h, ::stride_w] = x
            # 所以 backward 只取对应原输入位置的梯度。
            self.x.grad += dL_dX_up[
                :,
                :,
                ::self.stride[0],
                ::self.stride[1]
            ]
    def _get_inputs(self):
        res = [self.x, self.w]
        if self.b is not None: res.append(self.b)
        return res

# =========================================================================
# 6. 池化层家族 (MaxPool, MinPool, AvgPool)
# =========================================================================

class MaxPoolOp(Function):
    def __init__(self):
        super().__init__()
        self.x = None
        self.mask = None
        self.x_shape = None
        self.kernel_size = None
        self.stride = None
        self.padding = None

    def forward(self, x: Tensor, kernel_size, stride, padding=0):
        self.x = x
        self.x_shape = x.shape()
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        N, C, H, W = self.x_shape
        KH, KW = self.kernel_size

        col, _, _, out_h, out_w = _im2col_ext(
            x.data, KH, KW, stride=self.stride, padding=self.padding, xp=self.xp
        )
        col_reshaped = col.reshape(-1, C, KH * KW)
        
        # 寻找最大值及记录索引以用于梯度散布
        self.data = self.xp.max(col_reshaped, axis=2)
        self.mask = self.xp.argmax(col_reshaped, axis=2)
        
        self.data = self.data.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return Tensor(self.data, requires_grad=x.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
            
        if self.x.requires_grad:
            N, C, OH, OW = self.grad.shape
            dL_dO_flat = self.grad.transpose(0, 2, 3, 1).reshape(-1, C)
            KH, KW = self.kernel_size
            
            dL_dCol = self.xp.zeros((dL_dO_flat.shape[0], C, KH * KW), dtype=self.grad.dtype)
            
            # 使用高级索引进行梯度散布
            idx_range = self.xp.arange(dL_dO_flat.shape[0])[:, None]
            c_range = self.xp.arange(C)[None, :]
            dL_dCol[idx_range, c_range, self.mask] = dL_dO_flat
            
            dL_dX = _col2im_ext(
                dL_dCol.reshape(dL_dO_flat.shape[0], -1), 
                self.x_shape, KH, KW, 
                stride=self.stride, 
                padding=self.padding, 
                xp=self.xp
            )
            
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += dL_dX

    def _get_inputs(self):
        return [self.x]


class MinPoolOp(Function):
    def __init__(self):
        super().__init__()
        self.x = None
        self.mask = None
        self.x_shape = None
        self.kernel_size = None
        self.stride = None
        self.padding = None

    def forward(self, x: Tensor, kernel_size, stride, padding=0):
        self.x = x
        self.x_shape = x.shape()
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        N, C, H, W = self.x_shape
        KH, KW = self.kernel_size

        col, _, _, out_h, out_w = _im2col_ext(
            x.data, KH, KW, stride=self.stride, padding=self.padding, xp=self.xp
        )
        col_reshaped = col.reshape(-1, C, KH * KW)
        
        # 寻找最小值及记录索引
        self.data = self.xp.min(col_reshaped, axis=2)
        self.mask = self.xp.argmin(col_reshaped, axis=2)
        
        self.data = self.data.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return Tensor(self.data, requires_grad=x.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
            
        if self.x.requires_grad:
            N, C, OH, OW = self.grad.shape
            dL_dO_flat = self.grad.transpose(0, 2, 3, 1).reshape(-1, C)
            KH, KW = self.kernel_size
            
            dL_dCol = self.xp.zeros((dL_dO_flat.shape[0], C, KH * KW), dtype=self.grad.dtype)
            
            idx_range = self.xp.arange(dL_dO_flat.shape[0])[:, None]
            c_range = self.xp.arange(C)[None, :]
            dL_dCol[idx_range, c_range, self.mask] = dL_dO_flat
            
            dL_dX = _col2im_ext(
                dL_dCol.reshape(dL_dO_flat.shape[0], -1), 
                self.x_shape, KH, KW, 
                stride=self.stride, 
                padding=self.padding, 
                xp=self.xp
            )
            
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += dL_dX

    def _get_inputs(self):
        return [self.x]


class AvgPoolOp(Function):
    def __init__(self):
        super().__init__()
        self.x = None
        self.x_shape = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.pool_area = None

    def forward(self, x: Tensor, kernel_size, stride, padding=0):
        self.x = x
        self.x_shape = x.shape()
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        KH, KW = self.kernel_size
        self.pool_area = KH * KW
        
        N, C, H, W = self.x_shape
        
        col, _, _, out_h, out_w = _im2col_ext(
            x.data, KH, KW, stride=self.stride, padding=self.padding, xp=self.xp
        )
        
        # 计算均值
        col_reshaped = col.reshape(-1, C, self.pool_area)
        self.data = self.xp.mean(col_reshaped, axis=2)
        
        self.data = self.data.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return Tensor(self.data, requires_grad=x.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
            
        if self.x.requires_grad:
            N, C, OH, OW = self.grad.shape
            dL_dO_flat = self.grad.transpose(0, 2, 3, 1).reshape(-1, C)
            KH, KW = self.kernel_size
            
            # 均值池化的梯度是将梯度平均分配给窗口内的每个元素
            dL_dO_distributed = dL_dO_flat / self.pool_area
            dL_dCol = self.xp.repeat(dL_dO_distributed[:, :, None], self.pool_area, axis=2)
            
            dL_dX = _col2im_ext(
                dL_dCol.reshape(dL_dO_flat.shape[0], -1), 
                self.x_shape, KH, KW, 
                stride=self.stride, 
                padding=self.padding, 
                xp=self.xp
            )
            
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += dL_dX

    def _get_inputs(self):
        return [self.x]


# =========================================================================
# 7. 损失函数 (MSE, LogSoftmax, NLLLoss)
# =========================================================================

class MSE(Function):
    def forward(self, y_pred: Tensor, y_true: Tensor):
        self.y_pred = y_pred
        self.y_true = y_true
        
        diff = y_pred.data - y_true.data
        self.data = self.xp.mean(diff ** 2)
        
        return Tensor(self.data, requires_grad=y_pred.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
            
        if self.y_pred.requires_grad:
            N = self.y_pred.data.size
            grad_local = (2.0 / N) * (self.y_pred.data - self.y_true.data)
            
            if self.y_pred.grad is None:
                self.y_pred.grad = self.xp.zeros_like(self.y_pred.data)
            self.y_pred.grad += self.grad * grad_local

    def _get_inputs(self):
        return [self.y_pred, self.y_true]


class LogSoftmaxOp(Function):
    """
    负责 Log(Softmax(x)) 的数值稳定 Op 实现。
    """
    def forward(self, x: Tensor):
        self.x = x
        
        # 数值稳定性优化：减去最大值防止溢出
        max_x = self.x.data.max(axis=1, keepdims=True)
        stable_x = self.x.data - max_x
        exp_stable_x = self.xp.exp(stable_x)
        
        sum_exp = exp_stable_x.sum(axis=1, keepdims=True)
        log_sum_exp = self.xp.log(sum_exp)

        self.data = stable_x - log_sum_exp
        return Tensor(self.data, requires_grad=x.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
            
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
    """
    负对数似然损失 (NLL Loss)。
    通常与 LogSoftmax 配合使用以完成分类任务。
    """
    def forward(self, log_probs: Tensor, target: Tensor):
        self.log_probs = log_probs
        self.target = target
        
        N = log_probs.data.shape[0]
        # 使用高级索引提取正确类别的对数概率
        picked_log_probs = log_probs.data[self.xp.arange(N), target.data]
        self.data = -self.xp.mean(picked_log_probs)
        
        return Tensor(self.data, requires_grad=log_probs.requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
            
        if self.log_probs.requires_grad:
            N = self.log_probs.data.shape[0]
            grad_to_pass = self.xp.zeros_like(self.log_probs.data)
            
            # 在目标类别的位置写入梯度 -(1/N)
            grad_to_pass[self.xp.arange(N), self.target.data] = -1.0 / N
            grad_to_pass *= self.grad

            if self.log_probs.grad is None:
                self.log_probs.grad = self.xp.zeros_like(self.log_probs.data)
            self.log_probs.grad += grad_to_pass

    def _get_inputs(self):
        return [self.log_probs, self.target]

# =========================================================================
# 8. 归一化算子 (Normalization)
# =========================================================================

class BatchNorm2dOp(Function):
    """
    二维批量归一化算子 (BatchNorm2dOp)
    处理形状为 (N, C, H, W) 的图像特征数据。
    """
    def __init__(self, momentum=0.1, eps=1e-5, is_train=True):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.is_train = is_train

    def forward(self, x, gamma, beta, running_mean, running_var):
        self.x = x
        self.gamma = gamma
        self.beta = beta
        
        N, C, H, W = x.shape()
        
        if self.is_train:
            # 计算当前 batch 的均值和方差，针对 (N, H, W) 维度求聚合操作，保留 C 维度
            mean = self.xp.mean(x.data, axis=(0, 2, 3), keepdims=True)
            var = self.xp.var(x.data, axis=(0, 2, 3), keepdims=True)
            
            # 更新全局的 running_mean 和 running_var
            # 注意 running_mean 和 running_var 是作为 Tensor 传入的 (requires_grad=False)
            m = N * H * W
            unbiased_var = var * (m / (m - 1)) if m > 1 else var
            
            running_mean.data = (1 - self.momentum) * running_mean.data + self.momentum * mean
            running_var.data = (1 - self.momentum) * running_var.data + self.momentum * unbiased_var
            
            self.batch_mean = mean
            self.batch_var = var
        else:
            mean = running_mean.data
            var = running_var.data
            
        self.std_inv = 1.0 / self.xp.sqrt(var + self.eps)
        self.x_hat = (x.data - mean) * self.std_inv
        
        # 获取底层数据进行计算，gamma/beta 形状设定为 (1, C, 1, 1) 可直接广播
        gamma_data = gamma.data if gamma is not None else 1.0
        beta_data = beta.data if beta is not None else 0.0
        
        self.data = gamma_data * self.x_hat + beta_data
        
        requires_grad = x.requires_grad or (gamma is not None and gamma.requires_grad) or (beta is not None and beta.requires_grad)
        return Tensor(self.data, requires_grad=requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad
            
        N, C, H, W = self.x.shape()
        M = N * H * W
        
        gamma_data = self.gamma.data if self.gamma is not None else self.xp.ones((1, C, 1, 1))
        
        # 1. 计算 gamma 和 beta 的梯度
        if self.gamma is not None and self.gamma.requires_grad:
            dgamma = self.xp.sum(self.grad * self.x_hat, axis=(0, 2, 3), keepdims=True)
            if self.gamma.grad is None:
                self.gamma.grad = self.xp.zeros_like(self.gamma.data)
            self.gamma.grad += dgamma
            
        if self.beta is not None and self.beta.requires_grad:
            dbeta = self.xp.sum(self.grad, axis=(0, 2, 3), keepdims=True)
            if self.beta.grad is None:
                self.beta.grad = self.xp.zeros_like(self.beta.data)
            self.beta.grad += dbeta
            
        # 2. 计算输入的梯度
        if self.x.requires_grad:
            if self.is_train:
                # 训练模式下梯度的解析解
                dx_hat = self.grad * gamma_data
                dbeta_temp = self.xp.sum(dx_hat, axis=(0, 2, 3), keepdims=True)
                dgamma_temp = self.xp.sum(dx_hat * self.x_hat, axis=(0, 2, 3), keepdims=True)
                
                dx = (1.0 / M) * self.std_inv * (M * dx_hat - dbeta_temp - self.x_hat * dgamma_temp)
            else:
                # 评估模式下仅通过线性缩放传递梯度
                dx = self.grad * gamma_data * self.std_inv
                
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += dx

    def _get_inputs(self):
        return [self.x, self.gamma, self.beta]
# =========================================================================    
#算子融合
# =========================================================================
class FusedBatchNormReLUOp(Function):
    """
    真正 GPU kernel fusion 版 BatchNorm2d + ReLU。

    支持:
        - NCHW
        - float32
        - training / eval
        - gamma / beta
        - running_mean / running_var 更新
        - backward: dx, dgamma, dbeta

    forward:
        train:
            mean = mean(x)
            var = var(x)
            running_mean / running_var 更新
            y = ReLU(gamma * (x - mean) / sqrt(var + eps) + beta)

        eval:
            使用 running_mean / running_var
            y = ReLU(gamma * (x - running_mean) / sqrt(running_var + eps) + beta)

    backward:
        融合 ReLU backward + BN backward。
    """

    _cuda_forward_stats_kernel = None
    _cuda_forward_apply_kernel = None
    _cuda_backward_reduce_kernel = None
    _cuda_backward_dx_kernel = None

    def __init__(self, momentum=0.1, eps=1e-5, is_train=True):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.is_train = is_train

    # ============================================================
    # CUDA kernels
    # ============================================================

    @staticmethod
    def _get_forward_stats_kernel():
        if FusedBatchNormReLUOp._cuda_forward_stats_kernel is None:
            code = r'''
            extern "C" __global__
            void bn_relu_forward_stats(
                const float* x,
                float* running_mean,
                float* running_var,
                float* mean,
                float* inv_std,
                int N,
                int C,
                int H,
                int W,
                int M,
                float momentum,
                float eps,
                int is_train
            ) {
                int c = blockIdx.x;
                int tid = threadIdx.x;

                extern __shared__ float shared[];
                float* s_sum = shared;
                float* s_sumsq = shared + blockDim.x;

                float local_sum = 0.0f;
                float local_sumsq = 0.0f;

                if (is_train) {
                    for (int m = tid; m < M; m += blockDim.x) {
                        int n = m / (H * W);
                        int rem = m % (H * W);
                        int h = rem / W;
                        int w = rem % W;

                        int idx = ((n * C + c) * H + h) * W + w;
                        float v = x[idx];

                        local_sum += v;
                        local_sumsq += v * v;
                    }

                    s_sum[tid] = local_sum;
                    s_sumsq[tid] = local_sumsq;
                    __syncthreads();

                    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                        if (tid < stride) {
                            s_sum[tid] += s_sum[tid + stride];
                            s_sumsq[tid] += s_sumsq[tid + stride];
                        }
                        __syncthreads();
                    }

                    if (tid == 0) {
                        float mu = s_sum[0] / M;
                        float ex2 = s_sumsq[0] / M;
                        float var = ex2 - mu * mu;

                        if (var < 0.0f) {
                            var = 0.0f;
                        }

                        mean[c] = mu;
                        inv_std[c] = rsqrtf(var + eps);

                        float unbiased_var = var;
                        if (M > 1) {
                            unbiased_var = var * ((float)M / (float)(M - 1));
                        }

                        running_mean[c] =
                            (1.0f - momentum) * running_mean[c] + momentum * mu;

                        running_var[c] =
                            (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
                    }
                } else {
                    if (tid == 0) {
                        float mu = running_mean[c];
                        float var = running_var[c];

                        mean[c] = mu;
                        inv_std[c] = rsqrtf(var + eps);
                    }
                }
            }
            '''
            FusedBatchNormReLUOp._cuda_forward_stats_kernel = cp.RawKernel(
                code,
                "bn_relu_forward_stats"
            )

        return FusedBatchNormReLUOp._cuda_forward_stats_kernel

    @staticmethod
    def _get_forward_apply_kernel():
        if FusedBatchNormReLUOp._cuda_forward_apply_kernel is None:
            code = r'''
            extern "C" __global__
            void bn_relu_forward_apply(
                const float* x,
                const float* gamma,
                const float* beta,
                const float* mean,
                const float* inv_std,
                float* out,
                int total,
                int C,
                int H,
                int W
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx >= total) {
                    return;
                }

                int hw = H * W;
                int c = (idx / hw) % C;

                float x_hat = (x[idx] - mean[c]) * inv_std[c];
                float y = gamma[c] * x_hat + beta[c];

                out[idx] = y > 0.0f ? y : 0.0f;
            }
            '''
            FusedBatchNormReLUOp._cuda_forward_apply_kernel = cp.RawKernel(
                code,
                "bn_relu_forward_apply"
            )

        return FusedBatchNormReLUOp._cuda_forward_apply_kernel

    @staticmethod
    def _get_backward_reduce_kernel():
        if FusedBatchNormReLUOp._cuda_backward_reduce_kernel is None:
            code = r'''
            extern "C" __global__
            void bn_relu_backward_reduce(
                const float* x,
                const float* grad_out,
                const float* gamma,
                const float* beta,
                const float* mean,
                const float* inv_std,
                float* dgamma,
                float* dbeta,
                float* sum_dxhat,
                float* sum_dxhat_xhat,
                int N,
                int C,
                int H,
                int W,
                int M
            ) {
                int c = blockIdx.x;
                int tid = threadIdx.x;

                extern __shared__ float shared[];

                float* s_dgamma = shared;
                float* s_dbeta = shared + blockDim.x;
                float* s_sum_dxhat = shared + 2 * blockDim.x;
                float* s_sum_dxhat_xhat = shared + 3 * blockDim.x;

                float local_dgamma = 0.0f;
                float local_dbeta = 0.0f;
                float local_sum_dxhat = 0.0f;
                float local_sum_dxhat_xhat = 0.0f;

                float g = gamma[c];
                float be = beta[c];
                float mu = mean[c];
                float inv = inv_std[c];

                for (int m = tid; m < M; m += blockDim.x) {
                    int n = m / (H * W);
                    int rem = m % (H * W);
                    int h = rem / W;
                    int w = rem % W;

                    int idx = ((n * C + c) * H + h) * W + w;

                    float x_hat = (x[idx] - mu) * inv;
                    float y = g * x_hat + be;

                    float grad_bn = y > 0.0f ? grad_out[idx] : 0.0f;
                    float dxhat = grad_bn * g;

                    local_dgamma += grad_bn * x_hat;
                    local_dbeta += grad_bn;
                    local_sum_dxhat += dxhat;
                    local_sum_dxhat_xhat += dxhat * x_hat;
                }

                s_dgamma[tid] = local_dgamma;
                s_dbeta[tid] = local_dbeta;
                s_sum_dxhat[tid] = local_sum_dxhat;
                s_sum_dxhat_xhat[tid] = local_sum_dxhat_xhat;

                __syncthreads();

                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        s_dgamma[tid] += s_dgamma[tid + stride];
                        s_dbeta[tid] += s_dbeta[tid + stride];
                        s_sum_dxhat[tid] += s_sum_dxhat[tid + stride];
                        s_sum_dxhat_xhat[tid] += s_sum_dxhat_xhat[tid + stride];
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    dgamma[c] += s_dgamma[0];
                    dbeta[c] += s_dbeta[0];

                    sum_dxhat[c] = s_sum_dxhat[0];
                    sum_dxhat_xhat[c] = s_sum_dxhat_xhat[0];
                }
            }
            '''
            FusedBatchNormReLUOp._cuda_backward_reduce_kernel = cp.RawKernel(
                code,
                "bn_relu_backward_reduce"
            )

        return FusedBatchNormReLUOp._cuda_backward_reduce_kernel

    @staticmethod
    def _get_backward_dx_kernel():
        if FusedBatchNormReLUOp._cuda_backward_dx_kernel is None:
            code = r'''
            extern "C" __global__
            void bn_relu_backward_dx(
                const float* x,
                const float* grad_out,
                const float* gamma,
                const float* beta,
                const float* mean,
                const float* inv_std,
                const float* sum_dxhat,
                const float* sum_dxhat_xhat,
                float* dx,
                int total,
                int C,
                int H,
                int W,
                int M,
                int is_train
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx >= total) {
                    return;
                }

                int hw = H * W;
                int c = (idx / hw) % C;

                float x_hat = (x[idx] - mean[c]) * inv_std[c];
                float y = gamma[c] * x_hat + beta[c];

                float grad_bn = y > 0.0f ? grad_out[idx] : 0.0f;
                float dxhat = grad_bn * gamma[c];

                float grad_x;

                if (is_train) {
                    grad_x =
                        (inv_std[c] / (float)M) *
                        (
                            (float)M * dxhat
                            - sum_dxhat[c]
                            - x_hat * sum_dxhat_xhat[c]
                        );
                } else {
                    grad_x = dxhat * inv_std[c];
                }

                dx[idx] += grad_x;
            }
            '''
            FusedBatchNormReLUOp._cuda_backward_dx_kernel = cp.RawKernel(
                code,
                "bn_relu_backward_dx"
            )

        return FusedBatchNormReLUOp._cuda_backward_dx_kernel

    # ============================================================
    # Helpers
    # ============================================================

    def _is_cuda(self, x):
        return cp is not None and isinstance(x.data, cp.ndarray)

    def _ensure_float32_cuda(self, name, arr):
        if arr.dtype != cp.float32:
            raise TypeError(
                f"FusedBatchNormReLUOp CUDA 路径只支持 float32，"
                f"{name}.dtype={arr.dtype}"
            )

    def _as_grad_out_cuda(self):
        """
        把 self.grad 规范成 shape 与 self.x.data 完全一致的 cp.ndarray(float32)。

        关键原因：
        loss = y.sum() 的 backward 传下来的 grad 可能是：
            1. Python / NumPy 标量
            2. cp.ndarray 标量 shape=()
            3. cp.ndarray shape=(1,)
        但 RawKernel 里会按 grad_out[idx] 访问完整 NCHW，
        如果不扩展成完整 shape，就会越界读，导致 dgamma/dbeta 爆炸。
        """
        grad_out = self.grad

        if not isinstance(grad_out, cp.ndarray):
            grad_out = cp.asarray(grad_out, dtype=cp.float32)
        elif grad_out.dtype != cp.float32:
            grad_out = grad_out.astype(cp.float32)

        # 如果不是完整 NCHW shape，就尝试扩展
        expected_shape = self.x_data_contig.shape

        if grad_out.shape != expected_shape:
            if grad_out.size == 1:
                scalar = grad_out.reshape(-1)[0]
                grad_out = cp.full(expected_shape, scalar, dtype=cp.float32)
            else:
                grad_out = cp.broadcast_to(grad_out, expected_shape).astype(cp.float32)

        return cp.ascontiguousarray(grad_out)


    # ============================================================
    # Forward
    # ============================================================
    def _ensure_cuda_grad_buffer(self, tensor):
        """
        确保 tensor.grad 是合法的 cp.ndarray(float32)，并且 shape 正确。
        """
        if (
            tensor.grad is None
            or not isinstance(tensor.grad, cp.ndarray)
            or tensor.grad.shape != tensor.data.shape
            or tensor.grad.dtype != cp.float32
        ):
            tensor.grad = cp.zeros_like(tensor.data, dtype=cp.float32)

    def forward(self, x, gamma, beta, running_mean, running_var):
        self.x = x
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var

        N, C, H, W = x.shape()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.M = N * H * W
        total = N * C * H * W

        requires_grad = (
            x.requires_grad
            or gamma.requires_grad
            or beta.requires_grad
        )

        # --------------------------------------------------------
        # CUDA path
        # --------------------------------------------------------
        if self._is_cuda(x):
            self._ensure_float32_cuda("x", x.data)
            self._ensure_float32_cuda("gamma", gamma.data)
            self._ensure_float32_cuda("beta", beta.data)
            self._ensure_float32_cuda("running_mean", running_mean.data)
            self._ensure_float32_cuda("running_var", running_var.data)

            # RawKernel 使用 x[idx] 线性索引，必须保证 NCHW contiguous。
            # Conv2dOp 的输出经过 transpose(0, 3, 1, 2)，通常不是 contiguous。
            x_data = cp.ascontiguousarray(x.data)

            # 保存给 backward 使用，避免 backward 再错误读取非 contiguous 的 self.x.data
            self.x_data_contig = x_data

            gamma_data = cp.ascontiguousarray(gamma.data.reshape(-1))
            beta_data = cp.ascontiguousarray(beta.data.reshape(-1))

            # running_mean / running_var 需要被 kernel 原地更新，不能随便拷贝成独立 buffer。
            # 这里假设它们本身是 contiguous 的 (1, C, 1, 1)，reshape(-1) 是 view。
            running_mean_data = running_mean.data.reshape(-1)
            running_var_data = running_var.data.reshape(-1)

            self.mean = cp.empty((C,), dtype=cp.float32)
            self.inv_std = cp.empty((C,), dtype=cp.float32)

            # 输出也用 contiguous NCHW
            out = cp.empty(x_data.shape, dtype=cp.float32)
            block_reduce = 256
            shared_stats = block_reduce * 2 * 4

            stats_kernel = self._get_forward_stats_kernel()
            stats_kernel(
                (C,),
                (block_reduce,),
                (
                    x_data,
                    running_mean_data,
                    running_var_data,
                    self.mean,
                    self.inv_std,
                    N,
                    C,
                    H,
                    W,
                    self.M,
                    float(self.momentum),
                    float(self.eps),
                    int(self.is_train),
                ),
                shared_mem=shared_stats,
            )

            block = 256
            grid = ((total + block - 1) // block,)

            apply_kernel = self._get_forward_apply_kernel()
            apply_kernel(
                grid,
                (block,),
                (
                    x_data,
                    gamma_data,
                    beta_data,
                    self.mean,
                    self.inv_std,
                    out,
                    total,
                    C,
                    H,
                    W,
                )
            )

            self.data = out

            return Tensor(
                self.data,
                requires_grad=requires_grad,
                creator=self
            )

        # --------------------------------------------------------
        # CPU / NumPy fallback
        # --------------------------------------------------------
        if self.is_train:
            mean = self.xp.mean(x.data, axis=(0, 2, 3), keepdims=True)
            var = self.xp.var(x.data, axis=(0, 2, 3), keepdims=True)

            m = self.M
            unbiased_var = var * (m / (m - 1)) if m > 1 else var

            running_mean.data = (
                (1 - self.momentum) * running_mean.data
                + self.momentum * mean
            )

            running_var.data = (
                (1 - self.momentum) * running_var.data
                + self.momentum * unbiased_var
            )

            self.mean_cpu = mean
            self.var_cpu = var
        else:
            self.mean_cpu = running_mean.data
            self.var_cpu = running_var.data

        self.std_inv_cpu = 1.0 / self.xp.sqrt(self.var_cpu + self.eps)
        self.x_hat_cpu = (x.data - self.mean_cpu) * self.std_inv_cpu

        bn_out = gamma.data * self.x_hat_cpu + beta.data

        self.relu_mask_cpu = bn_out > 0
        self.data = self.xp.maximum(bn_out, 0)

        return Tensor(
            self.data,
            requires_grad=requires_grad,
            creator=self
        )

    # ============================================================
    # Backward
    # ============================================================

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        # --------------------------------------------------------
        # CUDA path
        # --------------------------------------------------------
        if self._is_cuda(self.x):
            grad_out = self._as_grad_out_cuda()

            x_data = self.x_data_contig

            gamma_data = cp.ascontiguousarray(self.gamma.data.reshape(-1))
            beta_data = cp.ascontiguousarray(self.beta.data.reshape(-1))
            C = self.C
            H = self.H
            W = self.W
            total = self.N * self.C * self.H * self.W

            if self.gamma.requires_grad and self.gamma.grad is None:
                self.gamma.grad = cp.zeros_like(self.gamma.data)

            if self.beta.requires_grad and self.beta.grad is None:
                self.beta.grad = cp.zeros_like(self.beta.data)

            # 如果 gamma / beta 不需要梯度，也给一个临时 buffer，方便 kernel 接口统一
            if self.gamma.requires_grad:
                self._ensure_cuda_grad_buffer(self.gamma)
                dgamma_buf = self.gamma.grad.reshape(-1)
            else:
                dgamma_buf = cp.zeros((C,), dtype=cp.float32)

            if self.beta.requires_grad:
                self._ensure_cuda_grad_buffer(self.beta)
                dbeta_buf = self.beta.grad.reshape(-1)
            else:
                dbeta_buf = cp.zeros((C,), dtype=cp.float32)
            sum_dxhat = cp.empty((C,), dtype=cp.float32)
            sum_dxhat_xhat = cp.empty((C,), dtype=cp.float32)

            block_reduce = 256
            shared_reduce = block_reduce * 4 * 4

            reduce_kernel = self._get_backward_reduce_kernel()
            reduce_kernel(
                (C,),
                (block_reduce,),
                (
                    x_data,
                    grad_out,
                    gamma_data,
                    beta_data,
                    self.mean,
                    self.inv_std,
                    dgamma_buf,
                    dbeta_buf,
                    sum_dxhat,
                    sum_dxhat_xhat,
                    self.N,
                    self.C,
                    self.H,
                    self.W,
                    self.M,
                ),
                shared_mem=shared_reduce,
            )

            if self.x.requires_grad:
                if (
                    self.x.grad is None
                    or not isinstance(self.x.grad, cp.ndarray)
                    or self.x.grad.shape != self.x.data.shape
                    or self.x.grad.dtype != cp.float32
                ):
                    self.x.grad = cp.zeros(self.x.data.shape, dtype=cp.float32)

                block = 256
                grid = ((total + block - 1) // block,)

                dx_kernel = self._get_backward_dx_kernel()
                dx_kernel(
                    grid,
                    (block,),
                    (
                        x_data,
                        grad_out,
                        gamma_data,
                        beta_data,
                        self.mean,
                        self.inv_std,
                        sum_dxhat,
                        sum_dxhat_xhat,
                        self.x.grad,
                        total,
                        self.C,
                        self.H,
                        self.W,
                        self.M,
                        int(self.is_train),
                    )
                )

            return

        # --------------------------------------------------------
        # CPU fallback
        # --------------------------------------------------------
        grad_bn = self.grad * self.relu_mask_cpu

        gamma_data = self.gamma.data

        if self.beta.requires_grad:
            d_beta = self.xp.sum(grad_bn, axis=(0, 2, 3), keepdims=True)
            if self.beta.grad is None:
                self.beta.grad = self.xp.zeros_like(self.beta.data)
            self.beta.grad += d_beta

        if self.gamma.requires_grad:
            d_gamma = self.xp.sum(
                grad_bn * self.x_hat_cpu,
                axis=(0, 2, 3),
                keepdims=True
            )
            if self.gamma.grad is None:
                self.gamma.grad = self.xp.zeros_like(self.gamma.data)
            self.gamma.grad += d_gamma

        if self.x.requires_grad:
            if self.is_train:
                dx_hat = grad_bn * gamma_data
                x_mu = self.x.data - self.mean_cpu
                M = self.M

                dvar = self.xp.sum(
                    dx_hat * x_mu * (-0.5) * (self.std_inv_cpu ** 3),
                    axis=(0, 2, 3),
                    keepdims=True
                )

                dmean = (
                    self.xp.sum(dx_hat * (-self.std_inv_cpu), axis=(0, 2, 3), keepdims=True)
                    + dvar * self.xp.mean(-2.0 * x_mu, axis=(0, 2, 3), keepdims=True)
                )

                dx = (
                    dx_hat * self.std_inv_cpu
                    + dvar * 2.0 * x_mu / M
                    + dmean / M
                )
            else:
                dx = grad_bn * gamma_data * self.std_inv_cpu

            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)

            self.x.grad += dx

    def _get_inputs(self):
        return [self.x, self.gamma, self.beta]

    def _get_inputs(self):
        parents = [self.x]

        if self.gamma is not None:
            parents.append(self.gamma)

        if self.beta is not None:
            parents.append(self.beta)

        return parents

class FusedBatchNormAddReLUOp(Function):
    """
    训练态 BatchNorm2d + residual Add + ReLU 融合算子。

    forward:
        y = ReLU(BN(x) + identity)

    backward:
        grad_identity = grad_out * mask
        grad_bn       = grad_out * mask
        然后走 BN backward 得到 dx / dgamma / dbeta

    用于 ResNet BasicBlock 第二段:
        conv2 -> bn2 -> Add(identity) -> relu2
    """

    _cuda_forward_stats_kernel = None
    _cuda_forward_apply_kernel = None
    _cuda_backward_reduce_kernel = None
    _cuda_backward_dx_identity_kernel = None

    def __init__(self, momentum=0.1, eps=1e-5, is_train=True):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.is_train = is_train

    # ============================================================
    # CUDA kernels
    # ============================================================
    @staticmethod
    def _get_forward_stats_kernel():
        """
        只计算当前 batch 的 mean / var / inv_std。
        不在 kernel 内更新 running_mean / running_var。

        这样避免复用 FusedBatchNormReLUOp 的 stats kernel 时，
        running stats 被间接更新或反推 var 造成不一致。
        """
        if FusedBatchNormAddReLUOp._cuda_forward_stats_kernel is None:
            code = r'''
            extern "C" __global__
            void bn_add_relu_forward_stats(
                const float* x,
                float* mean,
                float* var,
                float* inv_std,
                int N,
                int C,
                int H,
                int W,
                int M,
                float eps
            ) {
                int c = blockIdx.x;
                int tid = threadIdx.x;

                extern __shared__ float shared[];
                float* s_sum = shared;
                float* s_sumsq = shared + blockDim.x;

                float local_sum = 0.0f;
                float local_sumsq = 0.0f;

                for (int m = tid; m < M; m += blockDim.x) {
                    int n = m / (H * W);
                    int rem = m % (H * W);
                    int h = rem / W;
                    int w = rem % W;

                    int idx = ((n * C + c) * H + h) * W + w;
                    float v = x[idx];

                    local_sum += v;
                    local_sumsq += v * v;
                }

                s_sum[tid] = local_sum;
                s_sumsq[tid] = local_sumsq;
                __syncthreads();

                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        s_sum[tid] += s_sum[tid + stride];
                        s_sumsq[tid] += s_sumsq[tid + stride];
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    float mu = s_sum[0] / (float)M;
                    float ex2 = s_sumsq[0] / (float)M;
                    float v = ex2 - mu * mu;

                    if (v < 0.0f) {
                        v = 0.0f;
                    }

                    mean[c] = mu;
                    var[c] = v;
                    inv_std[c] = rsqrtf(v + eps);
                }
            }
            '''
            FusedBatchNormAddReLUOp._cuda_forward_stats_kernel = cp.RawKernel(
                code,
                "bn_add_relu_forward_stats"
            )

        return FusedBatchNormAddReLUOp._cuda_forward_stats_kernel
    @staticmethod
    def _get_forward_apply_kernel():
        if FusedBatchNormAddReLUOp._cuda_forward_apply_kernel is None:
            code = r'''
            extern "C" __global__
            void bn_add_relu_forward_apply(
                const float* x,
                const float* identity,
                const float* gamma,
                const float* beta,
                const float* mean,
                const float* inv_std,
                float* out,
                int total,
                int C,
                int H,
                int W
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= total) {
                    return;
                }

                int hw = H * W;
                int c = (idx / hw) % C;

                float x_hat = (x[idx] - mean[c]) * inv_std[c];
                float bn = gamma[c] * x_hat + beta[c];
                float z = bn + identity[idx];

                out[idx] = z > 0.0f ? z : 0.0f;
            }
            '''
            FusedBatchNormAddReLUOp._cuda_forward_apply_kernel = cp.RawKernel(
                code,
                "bn_add_relu_forward_apply"
            )

        return FusedBatchNormAddReLUOp._cuda_forward_apply_kernel

    @staticmethod
    def _get_backward_reduce_kernel():
        if FusedBatchNormAddReLUOp._cuda_backward_reduce_kernel is None:
            code = r'''
            extern "C" __global__
            void bn_add_relu_backward_reduce(
                const float* x,
                const float* identity,
                const float* grad_out,
                const float* gamma,
                const float* beta,
                const float* mean,
                const float* inv_std,
                float* dgamma,
                float* dbeta,
                float* sum_dxhat,
                float* sum_dxhat_xhat,
                int N,
                int C,
                int H,
                int W,
                int M
            ) {
                int c = blockIdx.x;
                int tid = threadIdx.x;

                extern __shared__ float shared[];

                float* s_dgamma = shared;
                float* s_dbeta = shared + blockDim.x;
                float* s_sum_dxhat = shared + 2 * blockDim.x;
                float* s_sum_dxhat_xhat = shared + 3 * blockDim.x;

                float local_dgamma = 0.0f;
                float local_dbeta = 0.0f;
                float local_sum_dxhat = 0.0f;
                float local_sum_dxhat_xhat = 0.0f;

                float g = gamma[c];
                float be = beta[c];
                float mu = mean[c];
                float inv = inv_std[c];

                for (int m = tid; m < M; m += blockDim.x) {
                    int n = m / (H * W);
                    int rem = m % (H * W);
                    int h = rem / W;
                    int w = rem % W;

                    int idx = ((n * C + c) * H + h) * W + w;

                    float x_hat = (x[idx] - mu) * inv;
                    float bn = g * x_hat + be;
                    float z = bn + identity[idx];

                    float grad_bn = z > 0.0f ? grad_out[idx] : 0.0f;
                    float dxhat = grad_bn * g;

                    local_dgamma += grad_bn * x_hat;
                    local_dbeta += grad_bn;
                    local_sum_dxhat += dxhat;
                    local_sum_dxhat_xhat += dxhat * x_hat;
                }

                s_dgamma[tid] = local_dgamma;
                s_dbeta[tid] = local_dbeta;
                s_sum_dxhat[tid] = local_sum_dxhat;
                s_sum_dxhat_xhat[tid] = local_sum_dxhat_xhat;

                __syncthreads();

                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        s_dgamma[tid] += s_dgamma[tid + stride];
                        s_dbeta[tid] += s_dbeta[tid + stride];
                        s_sum_dxhat[tid] += s_sum_dxhat[tid + stride];
                        s_sum_dxhat_xhat[tid] += s_sum_dxhat_xhat[tid + stride];
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    dgamma[c] += s_dgamma[0];
                    dbeta[c] += s_dbeta[0];

                    sum_dxhat[c] = s_sum_dxhat[0];
                    sum_dxhat_xhat[c] = s_sum_dxhat_xhat[0];
                }
            }
            '''
            FusedBatchNormAddReLUOp._cuda_backward_reduce_kernel = cp.RawKernel(
                code,
                "bn_add_relu_backward_reduce"
            )

        return FusedBatchNormAddReLUOp._cuda_backward_reduce_kernel

    @staticmethod
    def _get_backward_dx_identity_kernel():
        if FusedBatchNormAddReLUOp._cuda_backward_dx_identity_kernel is None:
            code = r'''
            extern "C" __global__
            void bn_add_relu_backward_dx_identity(
                const float* x,
                const float* identity,
                const float* grad_out,
                const float* gamma,
                const float* beta,
                const float* mean,
                const float* inv_std,
                const float* sum_dxhat,
                const float* sum_dxhat_xhat,
                float* dx,
                float* didentity,
                int total,
                int C,
                int H,
                int W,
                int M,
                int is_train
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= total) {
                    return;
                }

                int hw = H * W;
                int c = (idx / hw) % C;

                float x_hat = (x[idx] - mean[c]) * inv_std[c];
                float bn = gamma[c] * x_hat + beta[c];
                float z = bn + identity[idx];

                float grad_bn = z > 0.0f ? grad_out[idx] : 0.0f;

                // Add 分支梯度
                didentity[idx] += grad_bn;

                // BN 输入分支梯度
                float dxhat = grad_bn * gamma[c];
                float grad_x;

                if (is_train) {
                    grad_x =
                        (inv_std[c] / (float)M) *
                        (
                            (float)M * dxhat
                            - sum_dxhat[c]
                            - x_hat * sum_dxhat_xhat[c]
                        );
                } else {
                    grad_x = dxhat * inv_std[c];
                }

                dx[idx] += grad_x;
            }
            '''
            FusedBatchNormAddReLUOp._cuda_backward_dx_identity_kernel = cp.RawKernel(
                code,
                "bn_add_relu_backward_dx_identity"
            )

        return FusedBatchNormAddReLUOp._cuda_backward_dx_identity_kernel

    # ============================================================
    # Helpers
    # ============================================================

    def _is_cuda(self, x):
        return cp is not None and isinstance(x.data, cp.ndarray)

    def _ensure_float32_cuda(self, name, arr):
        if arr.dtype != cp.float32:
            raise TypeError(
                f"FusedBatchNormAddReLUOp CUDA 路径只支持 float32，"
                f"{name}.dtype={arr.dtype}"
            )

    def _ensure_cuda_grad_buffer(self, tensor):
        """
        RawKernel 用线性索引写 grad，因此 grad buffer 必须是 contiguous。
        """
        need_new = (
            tensor.grad is None
            or not isinstance(tensor.grad, cp.ndarray)
            or tensor.grad.shape != tensor.data.shape
            or tensor.grad.dtype != cp.float32
        )

        if not need_new:
            try:
                if not tensor.grad.flags.c_contiguous:
                    need_new = True
            except Exception:
                need_new = True

        if need_new:
            tensor.grad = cp.zeros(tensor.data.shape, dtype=cp.float32)

    def _as_grad_out_cuda(self):
        grad_out = self.grad

        if not isinstance(grad_out, cp.ndarray):
            grad_out = cp.asarray(grad_out, dtype=cp.float32)
        elif grad_out.dtype != cp.float32:
            grad_out = grad_out.astype(cp.float32)

        expected_shape = self.x_data_contig.shape

        if grad_out.shape != expected_shape:
            if grad_out.size == 1:
                scalar = grad_out.reshape(-1)[0]
                grad_out = cp.full(expected_shape, scalar, dtype=cp.float32)
            else:
                grad_out = cp.broadcast_to(grad_out, expected_shape).astype(cp.float32)

        return cp.ascontiguousarray(grad_out)

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x, identity, gamma, beta, running_mean, running_var):
        self.x = x
        self.identity = identity
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var

        if x.shape() != identity.shape():
            raise ValueError(
                f"FusedBatchNormAddReLUOp 要求 x 和 identity shape 相同，"
                f"但收到 x={x.shape()}, identity={identity.shape()}"
            )

        N, C, H, W = x.shape()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.M = N * H * W
        total = N * C * H * W

        requires_grad = (
            x.requires_grad
            or identity.requires_grad
            or gamma.requires_grad
            or beta.requires_grad
        )

        # --------------------------------------------------------
        # CUDA path
        # --------------------------------------------------------
        # --------------------------------------------------------
        # CUDA path
        # --------------------------------------------------------
        if self._is_cuda(x):
            self._ensure_float32_cuda("x", x.data)
            self._ensure_float32_cuda("identity", identity.data)
            self._ensure_float32_cuda("gamma", gamma.data)
            self._ensure_float32_cuda("beta", beta.data)
            self._ensure_float32_cuda("running_mean", running_mean.data)
            self._ensure_float32_cuda("running_var", running_var.data)

            # RawKernel 使用线性 idx 访问，必须保证 contiguous。
            x_data = cp.ascontiguousarray(x.data)
            identity_data = cp.ascontiguousarray(identity.data)

            self.x_data_contig = x_data
            self.identity_data_contig = identity_data

            gamma_data = cp.ascontiguousarray(gamma.data.reshape(-1))
            beta_data = cp.ascontiguousarray(beta.data.reshape(-1))

            running_mean_flat = cp.ascontiguousarray(running_mean.data.reshape(-1))
            running_var_flat = cp.ascontiguousarray(running_var.data.reshape(-1))

            self.mean = cp.empty((C,), dtype=cp.float32)
            self.var = cp.empty((C,), dtype=cp.float32)
            self.inv_std = cp.empty((C,), dtype=cp.float32)

            out = cp.empty(x_data.shape, dtype=cp.float32)

            # ====================================================
            # Train / Eval 分开处理
            # ====================================================
            if self.is_train:
                # 1. RawKernel 只计算 batch mean / var / inv_std
                block_reduce = 256
                shared_stats = block_reduce * 2 * 4

                stats_kernel = self._get_forward_stats_kernel()
                stats_kernel(
                    (C,),
                    (block_reduce,),
                    (
                        x_data,
                        self.mean,
                        self.var,
                        self.inv_std,
                        N,
                        C,
                        H,
                        W,
                        self.M,
                        float(self.eps),
                    ),
                    shared_mem=shared_stats,
                )

                # 2. 在 CuPy 层显式更新 running stats
                #    与 BatchNorm2dOp 保持同一公式：
                #    running_mean = (1 - momentum) * running_mean + momentum * mean
                #    running_var  = (1 - momentum) * running_var  + momentum * unbiased_var
                if self.M > 1:
                    unbiased_var = self.var * (float(self.M) / float(self.M - 1))
                else:
                    unbiased_var = self.var

                new_running_mean = (
                    (1.0 - float(self.momentum)) * running_mean_flat
                    + float(self.momentum) * self.mean
                )

                new_running_var = (
                    (1.0 - float(self.momentum)) * running_var_flat
                    + float(self.momentum) * unbiased_var
                )

                running_mean.data[...] = new_running_mean.reshape(running_mean.data.shape)
                running_var.data[...] = new_running_var.reshape(running_var.data.shape)

            else:
                # eval 阶段绝对不能计算 batch stats，也不能更新 running stats。
                self.mean[...] = running_mean_flat
                self.var[...] = running_var_flat
                self.inv_std[...] = 1.0 / cp.sqrt(self.var + cp.float32(self.eps))

            # 3. BN + Add + ReLU apply
            block = 256
            grid = ((total + block - 1) // block,)

            apply_kernel = self._get_forward_apply_kernel()
            apply_kernel(
                grid,
                (block,),
                (
                    x_data,
                    identity_data,
                    gamma_data,
                    beta_data,
                    self.mean,
                    self.inv_std,
                    out,
                    total,
                    C,
                    H,
                    W,
                )
            )

            self.data = out

            return Tensor(
                self.data,
                requires_grad=requires_grad,
                creator=self
            )
        
        # --------------------------------------------------------
        # CPU / NumPy fallback
        # --------------------------------------------------------
        if self.is_train:
            mean = self.xp.mean(x.data, axis=(0, 2, 3), keepdims=True)
            var = self.xp.var(x.data, axis=(0, 2, 3), keepdims=True)

            m = self.M
            unbiased_var = var * (m / (m - 1)) if m > 1 else var

            running_mean.data = (
                (1 - self.momentum) * running_mean.data
                + self.momentum * mean
            )

            running_var.data = (
                (1 - self.momentum) * running_var.data
                + self.momentum * unbiased_var
            )

            self.mean_cpu = mean
            self.var_cpu = var
        else:
            self.mean_cpu = running_mean.data
            self.var_cpu = running_var.data

        self.std_inv_cpu = 1.0 / self.xp.sqrt(self.var_cpu + self.eps)
        self.x_hat_cpu = (x.data - self.mean_cpu) * self.std_inv_cpu

        bn_out = gamma.data * self.x_hat_cpu + beta.data
        z = bn_out + identity.data

        self.relu_mask_cpu = z > 0
        self.data = self.xp.maximum(z, 0)

        return Tensor(
            self.data,
            requires_grad=requires_grad,
            creator=self
        )

    # ============================================================
    # Backward
    # ============================================================

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        # --------------------------------------------------------
        # CUDA path
        # --------------------------------------------------------
        if self._is_cuda(self.x):
            grad_out = self._as_grad_out_cuda()

            x_data = self.x_data_contig
            identity_data = self.identity_data_contig

            gamma_data = cp.ascontiguousarray(self.gamma.data.reshape(-1))
            beta_data = cp.ascontiguousarray(self.beta.data.reshape(-1))

            C = self.C
            total = self.N * self.C * self.H * self.W

            if self.gamma.requires_grad:
                self._ensure_cuda_grad_buffer(self.gamma)
                dgamma_buf = self.gamma.grad.reshape(-1)
            else:
                dgamma_buf = cp.zeros((C,), dtype=cp.float32)

            if self.beta.requires_grad:
                self._ensure_cuda_grad_buffer(self.beta)
                dbeta_buf = self.beta.grad.reshape(-1)
            else:
                dbeta_buf = cp.zeros((C,), dtype=cp.float32)

            sum_dxhat = cp.empty((C,), dtype=cp.float32)
            sum_dxhat_xhat = cp.empty((C,), dtype=cp.float32)

            block_reduce = 256
            shared_reduce = block_reduce * 4 * 4

            reduce_kernel = self._get_backward_reduce_kernel()
            reduce_kernel(
                (C,),
                (block_reduce,),
                (
                    x_data,
                    identity_data,
                    grad_out,
                    gamma_data,
                    beta_data,
                    self.mean,
                    self.inv_std,
                    dgamma_buf,
                    dbeta_buf,
                    sum_dxhat,
                    sum_dxhat_xhat,
                    self.N,
                    self.C,
                    self.H,
                    self.W,
                    self.M,
                ),
                shared_mem=shared_reduce,
            )

            if self.x.requires_grad:
                self._ensure_cuda_grad_buffer(self.x)
                dx_buf = self.x.grad
            else:
                dx_buf = cp.zeros_like(x_data)

            if self.identity.requires_grad:
                self._ensure_cuda_grad_buffer(self.identity)
                didentity_buf = self.identity.grad
            else:
                didentity_buf = cp.zeros_like(identity_data)

            block = 256
            grid = ((total + block - 1) // block,)

            dx_identity_kernel = self._get_backward_dx_identity_kernel()
            dx_identity_kernel(
                grid,
                (block,),
                (
                    x_data,
                    identity_data,
                    grad_out,
                    gamma_data,
                    beta_data,
                    self.mean,
                    self.inv_std,
                    sum_dxhat,
                    sum_dxhat_xhat,
                    dx_buf,
                    didentity_buf,
                    total,
                    self.C,
                    self.H,
                    self.W,
                    self.M,
                    int(self.is_train),
                )
            )

            return

        # --------------------------------------------------------
        # CPU fallback
        # --------------------------------------------------------
        grad_add = self.grad * self.relu_mask_cpu

        if self.identity.requires_grad:
            if self.identity.grad is None:
                self.identity.grad = self.xp.zeros_like(self.identity.data)
            self.identity.grad += grad_add

        gamma_data = self.gamma.data

        if self.beta.requires_grad:
            d_beta = self.xp.sum(grad_add, axis=(0, 2, 3), keepdims=True)
            if self.beta.grad is None:
                self.beta.grad = self.xp.zeros_like(self.beta.data)
            self.beta.grad += d_beta

        if self.gamma.requires_grad:
            d_gamma = self.xp.sum(
                grad_add * self.x_hat_cpu,
                axis=(0, 2, 3),
                keepdims=True
            )
            if self.gamma.grad is None:
                self.gamma.grad = self.xp.zeros_like(self.gamma.data)
            self.gamma.grad += d_gamma

        if self.x.requires_grad:
            if self.is_train:
                dx_hat = grad_add * gamma_data
                M = self.M

                sum_dxhat = self.xp.sum(dx_hat, axis=(0, 2, 3), keepdims=True)
                sum_dxhat_xhat = self.xp.sum(
                    dx_hat * self.x_hat_cpu,
                    axis=(0, 2, 3),
                    keepdims=True
                )

                dx = (
                    self.std_inv_cpu / M
                    * (M * dx_hat - sum_dxhat - self.x_hat_cpu * sum_dxhat_xhat)
                )
            else:
                dx = grad_add * gamma_data * self.std_inv_cpu

            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)
            self.x.grad += dx

    def _get_inputs(self):
        return [self.x, self.identity, self.gamma, self.beta]

class FusedAddReLUOp(Function):
    """
    真正 GPU kernel fusion 版 Add + ReLU。

    forward:
        y = ReLU(x1 + x2)

    backward:
        dx1 = grad_out * ((x1 + x2) > 0)
        dx2 = grad_out * ((x1 + x2) > 0)
    """

    _cuda_forward_kernel = None
    _cuda_backward_both_kernel = None
    _cuda_backward_one_kernel = None

    @staticmethod
    def _get_forward_kernel():
        if FusedAddReLUOp._cuda_forward_kernel is None:
            FusedAddReLUOp._cuda_forward_kernel = cp.ElementwiseKernel(
                "float32 x1, float32 x2",
                "float32 y",
                """
                float v = x1 + x2;
                y = v > 0.0f ? v : 0.0f;
                """,
                "fused_add_relu_forward"
            )
        return FusedAddReLUOp._cuda_forward_kernel

    @staticmethod
    def _get_backward_both_kernel():
        if FusedAddReLUOp._cuda_backward_both_kernel is None:
            FusedAddReLUOp._cuda_backward_both_kernel = cp.ElementwiseKernel(
                "float32 grad_out, float32 x1, float32 x2, "
                "float32 old_grad_x1, float32 old_grad_x2",
                "float32 new_grad_x1, float32 new_grad_x2",
                """
                float g = (x1 + x2) > 0.0f ? grad_out : 0.0f;
                new_grad_x1 = old_grad_x1 + g;
                new_grad_x2 = old_grad_x2 + g;
                """,
                "fused_add_relu_backward_both"
            )
        return FusedAddReLUOp._cuda_backward_both_kernel

    @staticmethod
    def _get_backward_one_kernel():
        if FusedAddReLUOp._cuda_backward_one_kernel is None:
            FusedAddReLUOp._cuda_backward_one_kernel = cp.ElementwiseKernel(
                "float32 grad_out, float32 x1, float32 x2, float32 old_grad",
                "float32 new_grad",
                """
                float g = (x1 + x2) > 0.0f ? grad_out : 0.0f;
                new_grad = old_grad + g;
                """,
                "fused_add_relu_backward_one"
            )
        return FusedAddReLUOp._cuda_backward_one_kernel

    def _cuda_grad_out(self):
        """
        把 self.grad 规范成 cp.ndarray(float32)，shape 与 x1 一致。
        防止 sum backward 传入 numpy.float64 scalar。
        """
        grad_out = self.grad

        if not isinstance(grad_out, cp.ndarray):
            grad_out = cp.asarray(grad_out, dtype=cp.float32)
        elif grad_out.dtype != cp.float32:
            grad_out = grad_out.astype(cp.float32)

        # 如果是标量梯度，扩展成与输入同 shape
        if grad_out.shape == ():
            grad_out = cp.full_like(self.x1.data, grad_out, dtype=cp.float32)

        return grad_out

    def _ensure_cuda_grad_buffer(self, tensor):
        """
        确保 tensor.grad 是 cp.ndarray(float32)，shape 正确。
        """
        if (
            tensor.grad is None
            or not isinstance(tensor.grad, cp.ndarray)
            or tensor.grad.shape != tensor.data.shape
            or tensor.grad.dtype != cp.float32
        ):
            tensor.grad = cp.zeros_like(tensor.data, dtype=cp.float32)

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

        if x1.shape() != x2.shape():
            raise ValueError(
                f"FusedAddReLUOp 当前只支持两个输入 shape 完全相同，"
                f"但收到 x1.shape={x1.shape()}, x2.shape={x2.shape()}"
            )

        requires_grad = x1.requires_grad or x2.requires_grad

        # CUDA 路径
        if cp is not None and isinstance(x1.data, cp.ndarray):
            if x1.data.dtype != cp.float32 or x2.data.dtype != cp.float32:
                raise TypeError(
                    "FusedAddReLUOp CUDA 路径当前只支持 float32，"
                    f"但收到 x1={x1.data.dtype}, x2={x2.data.dtype}"
                )

            kernel = self._get_forward_kernel()
            self.data = kernel(x1.data, x2.data)

        # CPU fallback
        else:
            add_out = x1.data + x2.data
            self.data = self.xp.maximum(add_out, 0)

        return Tensor(
            self.data,
            requires_grad=requires_grad,
            creator=self
        )

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        # CUDA 路径
        if cp is not None and isinstance(self.x1.data, cp.ndarray):
            grad_out = self._cuda_grad_out()

            if self.x1.requires_grad:
                self._ensure_cuda_grad_buffer(self.x1)

            if self.x2.requires_grad:
                self._ensure_cuda_grad_buffer(self.x2)

            if self.x1.requires_grad and self.x2.requires_grad:
                kernel = self._get_backward_both_kernel()

                new_grad_x1, new_grad_x2 = kernel(
                    grad_out,
                    self.x1.data,
                    self.x2.data,
                    self.x1.grad,
                    self.x2.grad
                )

                self.x1.grad = new_grad_x1
                self.x2.grad = new_grad_x2

            elif self.x1.requires_grad:
                kernel = self._get_backward_one_kernel()

                self.x1.grad = kernel(
                    grad_out,
                    self.x1.data,
                    self.x2.data,
                    self.x1.grad
                )

            elif self.x2.requires_grad:
                kernel = self._get_backward_one_kernel()

                self.x2.grad = kernel(
                    grad_out,
                    self.x1.data,
                    self.x2.data,
                    self.x2.grad
                )

        # CPU fallback
        else:
            mask = (self.x1.data + self.x2.data) > 0
            grad_add = self.grad * mask

            if self.x1.requires_grad:
                if self.x1.grad is None:
                    self.x1.grad = self.xp.zeros_like(self.x1.data)
                self.x1.grad += grad_add

            if self.x2.requires_grad:
                if self.x2.grad is None:
                    self.x2.grad = self.xp.zeros_like(self.x2.data)
                self.x2.grad += grad_add

    def _get_inputs(self):
        return [self.x1, self.x2]
    
class FusedCrossEntropyLossOp(Function):
    """
    Fused Cross Entropy Loss。

    融合:
        LogSoftmaxOp + NLLLossOp

    输入:
        logits: shape = (N, C)
        target: shape = (N,) 或 (N, 1)，类别索引

    forward:
        loss = mean(-log_softmax(logits)[range(N), target])

    backward:
        dlogits = softmax(logits)
        dlogits[range(N), target] -= 1
        dlogits /= N
    """

    def forward(self, logits: Tensor, target: Tensor):
        self.logits = logits
        self.target = target

        if logits.data.ndim != 2:
            raise ValueError(
                f"FusedCrossEntropyLossOp 当前只支持 logits 为 2D，"
                f"但收到 shape={logits.shape()}"
            )

        N, C = logits.data.shape
        self.N = N
        self.C = C

        # target 转成 int64，兼容 CPU/GPU
        target_data = target.data.reshape(-1).astype(self.xp.int64)

        if target_data.shape[0] != N:
            raise ValueError(
                f"target batch size 必须等于 logits batch size，"
                f"但 target={target_data.shape[0]}, logits={N}"
            )

        self.target_data = target_data

        # 数值稳定 softmax / log_softmax
        shifted = logits.data - self.xp.max(logits.data, axis=1, keepdims=True)
        exp_shifted = self.xp.exp(shifted)
        sum_exp = self.xp.sum(exp_shifted, axis=1, keepdims=True)

        self.softmax = exp_shifted / sum_exp
        log_probs = shifted - self.xp.log(sum_exp)

        batch_indices = self.xp.arange(N)
        nll = -log_probs[batch_indices, target_data]

        self.data = self.xp.mean(nll)

        return Tensor(
            self.data,
            requires_grad=logits.requires_grad,
            creator=self
        )

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        if self.logits.requires_grad:
            grad_logits = self.softmax.copy()

            batch_indices = self.xp.arange(self.N)
            grad_logits[batch_indices, self.target_data] -= 1.0

            grad_logits = grad_logits / self.N

            if self.logits.grad is None:
                self.logits.grad = self.xp.zeros_like(self.logits.data)

            self.logits.grad += self.grad * grad_logits

    def _get_inputs(self):
        return [self.logits, self.target]
    
class FusedLinearReLUOp(Function):
    """
    Fused Linear + ReLU。

    融合:
        MatMul -> Add -> ReLU

    forward:
        y = ReLU(x @ weight + bias)

    当前主要支持 2D 输入:
        x:      (N, in_features)
        weight: (in_features, out_features)
        bias:   (1, out_features)
    """

    def forward(self, x: Tensor, weight: Tensor, bias: Tensor):
        self.x = x
        self.weight = weight
        self.bias = bias

        if x.data.ndim != 2:
            raise ValueError(
                f"FusedLinearReLUOp 当前只支持 2D 输入，"
                f"但收到 x.shape={x.shape()}"
            )

        linear_out = self.xp.matmul(x.data, weight.data)

        if bias is not None:
            linear_out = linear_out + bias.data

        self.relu_mask = linear_out > 0
        self.data = self.xp.maximum(linear_out, 0)

        requires_grad = (
            x.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )

        return Tensor(
            self.data,
            requires_grad=requires_grad,
            creator=self
        )

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        grad_linear = self.grad * self.relu_mask

        # x 梯度
        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)

            self.x.grad += self.xp.matmul(
                grad_linear,
                self.weight.data.swapaxes(-1, -2)
            )

        # weight 梯度
        if self.weight.requires_grad:
            if self.weight.grad is None:
                self.weight.grad = self.xp.zeros_like(self.weight.data)

            self.weight.grad += self.xp.matmul(
                self.x.data.swapaxes(-1, -2),
                grad_linear
            )

        # bias 梯度
        if self.bias is not None and self.bias.requires_grad:
            if self.bias.grad is None:
                self.bias.grad = self.xp.zeros_like(self.bias.data)

            db = self.xp.sum(grad_linear, axis=0, keepdims=True)

            # bias shape 一般是 (1, out_features)
            self.bias.grad += db.reshape(self.bias.data.shape)

    def _get_inputs(self):
        inputs = [self.x, self.weight]
        if self.bias is not None:
            inputs.append(self.bias)
        return inputs
    
class FusedMSELossOp(Function):
    """
    GPU fused MSELoss.

    forward:
        loss = mean((pred - target)^2)

    backward:
        d_pred   = grad_scale * 2 * (pred - target) / numel
        d_target = -d_pred

    说明：
        forward 用 RawKernel 做 reduction；
        backward 用 ElementwiseKernel，更稳，避免 RawKernel 写 grad buffer 失败。
    """

    _cuda_forward_kernel = None
    _cuda_backward_pred_kernel = None
    _cuda_backward_both_kernel = None

    def __init__(self, reduction="mean"):
        super().__init__()

        if reduction != "mean":
            raise NotImplementedError(
                "FusedMSELossOp 当前只支持 reduction='mean'"
            )

        self.reduction = reduction

    @staticmethod
    def _get_forward_kernel():
        if FusedMSELossOp._cuda_forward_kernel is None:
            code = r'''
            extern "C" __global__
            void fused_mse_forward(
                const float* pred,
                const float* target,
                float* loss,
                int numel
            ) {
                int tid = threadIdx.x;

                extern __shared__ float sdata[];

                float local_sum = 0.0f;

                for (int i = tid; i < numel; i += blockDim.x) {
                    float d = pred[i] - target[i];
                    local_sum += d * d;
                }

                sdata[tid] = local_sum;
                __syncthreads();

                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        sdata[tid] += sdata[tid + stride];
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    loss[0] = sdata[0] / (float)numel;
                }
            }
            '''

            FusedMSELossOp._cuda_forward_kernel = cp.RawKernel(
                code,
                "fused_mse_forward"
            )

        return FusedMSELossOp._cuda_forward_kernel

    @staticmethod
    def _get_backward_pred_kernel():
        if FusedMSELossOp._cuda_backward_pred_kernel is None:
            FusedMSELossOp._cuda_backward_pred_kernel = cp.ElementwiseKernel(
                "float32 pred, float32 target, float32 old_grad, float32 grad_scale, float32 inv_numel",
                "float32 new_grad",
                """
                new_grad = old_grad + grad_scale * 2.0f * (pred - target) * inv_numel;
                """,
                "fused_mse_backward_pred"
            )

        return FusedMSELossOp._cuda_backward_pred_kernel

    @staticmethod
    def _get_backward_both_kernel():
        if FusedMSELossOp._cuda_backward_both_kernel is None:
            FusedMSELossOp._cuda_backward_both_kernel = cp.ElementwiseKernel(
                "float32 pred, float32 target, float32 old_grad_pred, float32 old_grad_target, float32 grad_scale, float32 inv_numel",
                "float32 new_grad_pred, float32 new_grad_target",
                """
                float g = grad_scale * 2.0f * (pred - target) * inv_numel;
                new_grad_pred = old_grad_pred + g;
                new_grad_target = old_grad_target - g;
                """,
                "fused_mse_backward_both"
            )

        return FusedMSELossOp._cuda_backward_both_kernel

    def _is_cuda(self, tensor):
        return cp is not None and isinstance(tensor.data, cp.ndarray)

    def _ensure_float32_cuda(self, name, arr):
        if arr.dtype != cp.float32:
            raise TypeError(
                f"FusedMSELossOp CUDA 路径只支持 float32，"
                f"{name}.dtype={arr.dtype}"
            )

    def _ensure_cuda_grad_buffer(self, tensor):
        if (
            tensor.grad is None
            or not isinstance(tensor.grad, cp.ndarray)
            or tensor.grad.shape != tensor.data.shape
            or tensor.grad.dtype != cp.float32
        ):
            tensor.grad = cp.zeros(tensor.data.shape, dtype=cp.float32)

    def _grad_scale_float(self):
        """
        loss.backward() 的上游梯度通常是 1。
        这里统一转成 Python float。
        """
        grad = self.grad

        if grad is None:
            return 1.0

        if cp is not None and isinstance(grad, cp.ndarray):
            return float(grad.reshape(-1)[0].get())

        if isinstance(grad, np.ndarray):
            return float(grad.reshape(-1)[0])

        return float(grad)

    def _grad_scale_cuda(self):
        grad = self.grad

        if grad is None:
            return cp.float32(1.0)

        if isinstance(grad, cp.ndarray):
            return grad.astype(cp.float32, copy=False).reshape(())

        return cp.float32(self._grad_scale_float())

    def forward(self, y_pred: Tensor, y_true: Tensor):
        self.y_pred = y_pred
        self.y_true = y_true

        if y_pred.shape() != y_true.shape():
            raise ValueError(
                f"FusedMSELossOp 要求 pred 和 target shape 相同，"
                f"但收到 pred={y_pred.shape()}, target={y_true.shape()}"
            )

        requires_grad = y_pred.requires_grad or y_true.requires_grad

        # CUDA path
        if self._is_cuda(y_pred):
            self._ensure_float32_cuda("y_pred", y_pred.data)
            self._ensure_float32_cuda("y_true", y_true.data)

            self.pred_data_contig = cp.ascontiguousarray(y_pred.data)
            self.target_data_contig = cp.ascontiguousarray(y_true.data)
            self.numel = int(self.pred_data_contig.size)

            loss_buf = cp.empty((1,), dtype=cp.float32)

            block = 256
            shared_mem = block * 4

            kernel = self._get_forward_kernel()
            kernel(
                (1,),
                (block,),
                (
                    self.pred_data_contig,
                    self.target_data_contig,
                    loss_buf,
                    self.numel,
                ),
                shared_mem=shared_mem
            )

            # 这里保持 shape=(1,) 更稳，避免 0-d scalar 在 autograd 中处理异常
            self.data = loss_buf

            return Tensor(
                self.data,
                requires_grad=requires_grad,
                creator=self
            )

        # CPU fallback
        self.diff_cpu = y_pred.data - y_true.data
        self.numel = self.diff_cpu.size
        self.data = self.xp.mean(self.diff_cpu ** 2)

        return Tensor(
            self.data,
            requires_grad=requires_grad,
            creator=self
        )

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        # CUDA path
        if self._is_cuda(self.y_pred):
            grad_scale = self._grad_scale_cuda()
            inv_numel = cp.float32(1.0 / float(self.numel))

            if self.y_pred.requires_grad:
                self._ensure_cuda_grad_buffer(self.y_pred)

            if self.y_true.requires_grad:
                self._ensure_cuda_grad_buffer(self.y_true)

            if self.y_pred.requires_grad and self.y_true.requires_grad:
                kernel = self._get_backward_both_kernel()

                new_grad_pred, new_grad_target = kernel(
                    self.pred_data_contig,
                    self.target_data_contig,
                    self.y_pred.grad,
                    self.y_true.grad,
                    grad_scale,
                    inv_numel
                )

                self.y_pred.grad = new_grad_pred
                self.y_true.grad = new_grad_target

            elif self.y_pred.requires_grad:
                kernel = self._get_backward_pred_kernel()

                self.y_pred.grad = kernel(
                    self.pred_data_contig,
                    self.target_data_contig,
                    self.y_pred.grad,
                    grad_scale,
                    inv_numel
                )

            elif self.y_true.requires_grad:
                # 一般 DonkeyCar 的 target 不需要梯度，这个分支只是完整性支持。
                self._ensure_cuda_grad_buffer(self.y_true)

                # target-only 情况可以用 both kernel 的思想，但这里简单处理。
                g = grad_scale * 2.0 * (self.pred_data_contig - self.target_data_contig) * inv_numel
                self.y_true.grad -= g

            return

        # CPU fallback
        grad_scale = self._grad_scale_float()
        grad_local = grad_scale * (2.0 / self.numel) * self.diff_cpu

        if self.y_pred.requires_grad:
            if self.y_pred.grad is None:
                self.y_pred.grad = self.xp.zeros_like(self.y_pred.data)
            self.y_pred.grad += grad_local

        if self.y_true.requires_grad:
            if self.y_true.grad is None:
                self.y_true.grad = self.xp.zeros_like(self.y_true.data)
            self.y_true.grad -= grad_local

    def _get_inputs(self):
        return [self.y_pred, self.y_true]

class FusedConv2dReLUOp(Function):
    """
    静态融合版 Conv2d + ReLU。

    关键点：
    1. 不使用 naive direct convolution。
    2. 卷积主体复用 im2col / im2col_ext + GEMM。
    3. forward 中直接完成 Conv + Bias + ReLU。
    4. backward 中先融合 ReLU backward，再执行 Conv backward。
    """

    def __init__(self):
        super().__init__()
        self.x = None
        self.w = None
        self.b = None
        self.stride = None
        self.padding = None
        self.dilation = None
        self.groups = None
        self.col = None
        self.x_shape = None
        self.out_h = None
        self.out_w = None
        self.relu_mask = None

    def forward(self, x: Tensor, w: Tensor, b: Tensor = None,
                stride=1, padding=0, dilation=1, groups=1):
        self.x = x
        self.w = w
        self.b = b

        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups
        self.x_shape = x.shape()

        FN, C_group, KH, KW = w.shape()
        N, C_total, H, W_in = self.x_shape

        if groups <= 0 or C_total % groups != 0 or FN % groups != 0:
            raise ValueError("Groups 参数设置非法或输入输出通道数无法被整除")

        if C_group != C_total // groups:
            raise ValueError(
                f"权重输入通道与 groups 不匹配："
                f"期望 {C_total // groups}，实际得到 {C_group}"
            )

        use_fast_path = (
            self.dilation == (1, 1)
            and self.stride[0] == self.stride[1]
            and self.padding[0] == self.padding[1]
        )

        if use_fast_path:
            col, _, _ = _im2col(
                x.data,
                KH,
                KW,
                stride=self.stride[0],
                padding=self.padding[0],
                xp=self.xp
            )
            out_h = (H + 2 * self.padding[0] - KH) // self.stride[0] + 1
            out_w = (W_in + 2 * self.padding[0] - KW) // self.stride[0] + 1
        else:
            col, _, _, out_h, out_w = _im2col_ext(
                x.data,
                KH,
                KW,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                xp=self.xp
            )

        out_flat, self.col = _conv2d_gemm_forward(
            col=col,
            w_data=w.data,
            groups=groups,
            FN=FN,
            C_group=C_group,
            KH=KH,
            KW=KW,
            xp=self.xp
        )
        if b is not None:
            if b.data.size != FN:
                raise ValueError(
                    f"Bias 元素数量必须等于 out_channels={FN}，"
                    f"实际传入了 {b.data.size}。"
                )
            out_flat += b.data.reshape(1, FN)

        conv_out = out_flat.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        # 融合 ReLU forward
        self.relu_mask = conv_out > 0
        self.data = self.xp.maximum(conv_out, 0)

        self.out_h = out_h
        self.out_w = out_w

        requires_grad = (
            x.requires_grad
            or w.requires_grad
            or (b is not None and b.requires_grad)
        )

        return Tensor(self.data, requires_grad=requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        # 融合 ReLU backward
        dL_dO = self.grad * self.relu_mask

        FN, C_group, KH, KW = self.w.shape()
        groups = self.groups

        dL_dO_flat = dL_dO.transpose(0, 2, 3, 1).reshape(-1, FN)

        # bias 梯度
        if self.b is not None and self.b.requires_grad:
            dL_dB = self.xp.sum(dL_dO_flat, axis=0, keepdims=True)

            if self.b.grad is None:
                self.b.grad = self.xp.zeros_like(self.b.data)

            self.b.grad += dL_dB.reshape(self.b.shape())

        need_w_grad = self.w.requires_grad
        need_x_grad = self.x.requires_grad
        dL_dW = dL_dCol = None

        if need_w_grad or need_x_grad:
            dL_dW, dL_dCol = _conv2d_gemm_backward_select(
                col=self.col,
                dO_flat=dL_dO_flat,
                w_data=self.w.data,
                groups=groups,
                FN=FN,
                C_group=C_group,
                KH=KH,
                KW=KW,
                xp=self.xp,
                need_w=need_w_grad,
                need_col=need_x_grad,
            )

        # weight 梯度
        if need_w_grad:
            if self.w.grad is None:
                self.w.grad = self.xp.zeros_like(self.w.data)

            self.w.grad += dL_dW

        # input 梯度
        if need_x_grad:
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)

            use_fast_path = (
                self.dilation == (1, 1)
                and self.stride[0] == self.stride[1]
                and self.padding[0] == self.padding[1]
            )

            if use_fast_path:
                self.x.grad += _col2im(
                    dL_dCol,
                    self.x_shape,
                    KH,
                    KW,
                    stride=self.stride[0],
                    padding=self.padding[0],
                    xp=self.xp
                )
            else:
                self.x.grad += _col2im_ext(
                    dL_dCol,
                    self.x_shape,
                    KH,
                    KW,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    xp=self.xp
                )

    def _get_inputs(self):
        parents = [self.x, self.w]
        if self.b is not None:
            parents.append(self.b)
        return parents
    
class FusedConvBNReLUOp(Function):
    """
    训练态 Conv2d + BatchNorm2d + ReLU 融合算子。

    forward:
        conv = Conv2d(x, w, b)
        bn   = gamma * (conv - mean) / sqrt(var + eps) + beta
        out  = ReLU(bn)

    backward:
        grad_out -> ReLU backward -> BN backward -> Conv backward

    当前版本复用 im2col + GEMM 卷积路径。
    """

    _profile_events = []

    @classmethod
    def reset_profile_events(cls):
        FusedConvBNReLUOp._profile_events.clear()

    @classmethod
    def profile_events(cls):
        return list(FusedConvBNReLUOp._profile_events)

    @classmethod
    def profile_summary(cls, top_n=10):
        events = FusedConvBNReLUOp.profile_events()
        if not events:
            return "[JIT fused profile] no fused conv/bn stage profile has been recorded."

        by_op_phase = {}
        for event in events:
            key = (event["op_name"], event["phase"])
            item = by_op_phase.setdefault(key, {"calls": 0, "stages": {}})
            item["calls"] += 1
            for stage_name, ms in event["stages"].items():
                item["stages"][stage_name] = item["stages"].get(stage_name, 0.0) + ms

        lines = ["[JIT fused profile] internal stages by op/phase:"]
        for (op_name, phase), item in sorted(
            by_op_phase.items(),
            key=lambda kv: sum(kv[1]["stages"].values()),
            reverse=True,
        ):
            stage_text = ", ".join(
                f"{name}={ms:.3f} ms"
                for name, ms in sorted(item["stages"].items(), key=lambda kv: kv[1], reverse=True)
            )
            lines.append(f"  {op_name}.{phase}: calls={item['calls']}, {stage_text}")

        lines.append(f"[JIT fused profile] slowest fused op events top {top_n}:")
        for event in sorted(
            events,
            key=lambda item: sum(item["stages"].values()),
            reverse=True,
        )[:top_n]:
            total = sum(event["stages"].values())
            stage_text = ", ".join(
                f"{name}={ms:.3f}"
                for name, ms in sorted(event["stages"].items(), key=lambda kv: kv[1], reverse=True)
            )
            lines.append(
                f"  {event['profile_name']} {event['op_name']}.{event['phase']}: "
                f"total={total:.3f} ms; {stage_text}"
            )
        return "\n".join(lines)

    def __init__(self, momentum=0.1, eps=1e-5, is_train=True, profile=False, profile_name=None):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.is_train = is_train
        self.profile = profile
        self.profile_name = profile_name or self.__class__.__name__
        self.profile_stats = {"forward": {}, "backward": {}}

    def _profile_sync(self):
        if self.profile and GPU_AVAILABLE and cp is not None and self.xp is cp:
            cp.cuda.Stream.null.synchronize()

    def _profile_start(self):
        if not self.profile:
            return None
        self._profile_sync()
        return time.perf_counter()

    def _profile_end(self, phase, stage_name, start):
        if start is None:
            return
        self._profile_sync()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        phase_stats = self.profile_stats.setdefault(phase, {})
        phase_stats[stage_name] = phase_stats.get(stage_name, 0.0) + elapsed_ms

    def _profile_record_event(self, phase):
        if not self.profile:
            return
        FusedConvBNReLUOp._profile_events.append(
            {
                "profile_name": self.profile_name,
                "op_name": self.__class__.__name__,
                "phase": phase,
                "stages": dict(self.profile_stats.get(phase, {})),
            }
        )

    def forward(
        self,
        x: Tensor,
        w: Tensor,
        b: Tensor,
        gamma: Tensor,
        beta: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        self.x = x
        self.w = w
        self.b = b
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var

        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups
        self.x_shape = x.shape()

        FN, C_group, KH, KW = w.shape()
        N, C_total, H, W_in = self.x_shape

        if groups <= 0 or C_total % groups != 0 or FN % groups != 0:
            raise ValueError("Groups 参数设置非法或输入输出通道数无法被整除")

        if C_group != C_total // groups:
            raise ValueError(
                f"权重输入通道与 groups 不匹配：期望 {C_total // groups}，实际得到 {C_group}"
            )

        use_fast_path = (
            self.dilation == (1, 1)
            and self.stride[0] == self.stride[1]
            and self.padding[0] == self.padding[1]
        )

        stage_t0 = self._profile_start()
        if use_fast_path:
            col, _, _ = _im2col(
                x.data,
                KH,
                KW,
                stride=self.stride[0],
                padding=self.padding[0],
                xp=self.xp,
            )
            out_h = (H + 2 * self.padding[0] - KH) // self.stride[0] + 1
            out_w = (W_in + 2 * self.padding[0] - KW) // self.stride[0] + 1
        else:
            col, _, _, out_h, out_w = _im2col_ext(
                x.data,
                KH,
                KW,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                xp=self.xp,
            )
        self._profile_end("forward", "im2col_ms", stage_t0)

        stage_t0 = self._profile_start()
        conv_flat, self.col = _conv2d_gemm_forward(
            col=col,
            w_data=w.data,
            groups=groups,
            FN=FN,
            C_group=C_group,
            KH=KH,
            KW=KW,
            xp=self.xp
        )
        self._profile_end("forward", "gemm_ms", stage_t0)

        stage_t0 = self._profile_start()
        if b is not None:
            if b.data.size != FN:
                raise ValueError(
                    f"Bias 元素数量必须等于 out_channels={FN}，实际传入 {b.data.size}"
                )
            conv_flat += b.data.reshape(1, FN)
        self._profile_end("forward", "bias_ms", stage_t0)

        stage_t0 = self._profile_start()
        conv_out = conv_flat.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        self._profile_end("forward", "reshape_ms", stage_t0)

        self.out_h = out_h
        self.out_w = out_w

        # ----------------------------
        # BatchNorm forward
        # ----------------------------
        self.M = N * out_h * out_w

        if self.is_train:
            stage_t0 = self._profile_start()
            mean = self.xp.mean(conv_out, axis=(0, 2, 3), keepdims=True)
            self._profile_end("forward", "bn_mean_ms", stage_t0)

            stage_t0 = self._profile_start()
            x_hat = conv_out
            x_hat -= mean
            var = self.xp.mean(x_hat * x_hat, axis=(0, 2, 3), keepdims=True)
            self._profile_end("forward", "bn_var_ms", stage_t0)

            stage_t0 = self._profile_start()
            unbiased_var = var * (self.M / (self.M - 1)) if self.M > 1 else var

            running_mean.data = (
                (1.0 - self.momentum) * running_mean.data
                + self.momentum * mean
            )

            running_var.data = (
                (1.0 - self.momentum) * running_var.data
                + self.momentum * unbiased_var
            )

            self.mean = mean
            self.var = var
            self._profile_end("forward", "running_stats_ms", stage_t0)
        else:
            stage_t0 = self._profile_start()
            self.mean = running_mean.data
            self.var = running_var.data
            x_hat = conv_out
            x_hat -= self.mean
            self._profile_end("forward", "bn_eval_center_ms", stage_t0)

        stage_t0 = self._profile_start()
        self.std_inv = 1.0 / self.xp.sqrt(self.var + self.eps)
        x_hat *= self.std_inv
        self.x_hat = x_hat

        self.data = self.x_hat * gamma.data
        self.data += beta.data

        # ----------------------------
        # ReLU forward
        # ----------------------------
        self.relu_mask = self.data > 0
        self.xp.maximum(self.data, 0, out=self.data)
        self._profile_end("forward", "affine_relu_ms", stage_t0)

        requires_grad = (
            x.requires_grad
            or w.requires_grad
            or (b is not None and b.requires_grad)
            or gamma.requires_grad
            or beta.requires_grad
        )

        self._profile_record_event("forward")
        return Tensor(self.data, requires_grad=requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        # ----------------------------
        # ReLU backward
        # ----------------------------
        stage_t0 = self._profile_start()
        grad_bn = self.grad * self.relu_mask
        self._profile_end("backward", "relu_mask_ms", stage_t0)
        self._backward_bn_conv(grad_bn)
        self._profile_record_event("backward")

    def _backward_bn_conv(self, grad_bn):
        FN, C_group, KH, KW = self.w.shape()
        groups = self.groups

        # ----------------------------
        # BN backward
        # ----------------------------
        stage_t0 = self._profile_start()
        if self.beta.requires_grad:
            d_beta = self.xp.sum(grad_bn, axis=(0, 2, 3), keepdims=True)
            if self.beta.grad is None:
                self.beta.grad = self.xp.zeros_like(self.beta.data)
            self.beta.grad += d_beta

        if self.gamma.requires_grad:
            d_gamma = self.xp.sum(
                grad_bn * self.x_hat,
                axis=(0, 2, 3),
                keepdims=True,
            )
            if self.gamma.grad is None:
                self.gamma.grad = self.xp.zeros_like(self.gamma.data)
            self.gamma.grad += d_gamma
        self._profile_end("backward", "bn_param_grad_ms", stage_t0)

        need_b_grad = self.b is not None and self.b.requires_grad
        need_w_grad = self.w.requires_grad
        need_x_grad = self.x.requires_grad

        if not (need_b_grad or need_w_grad or need_x_grad):
            return

        gamma_data = self.gamma.data

        if self.is_train:
            stage_t0 = self._profile_start()
            grad_conv = grad_bn
            grad_conv *= gamma_data
            self._profile_end("backward", "bn_gamma_scale_ms", stage_t0)

            stage_t0 = self._profile_start()
            sum_dx_hat = self.xp.sum(grad_conv, axis=(0, 2, 3), keepdims=True)
            sum_dx_hat_xhat = self.xp.sum(
                grad_conv * self.x_hat,
                axis=(0, 2, 3),
                keepdims=True,
            )
            self._profile_end("backward", "bn_reduce_ms", stage_t0)

            stage_t0 = self._profile_start()
            grad_conv *= self.M
            grad_conv -= sum_dx_hat
            grad_conv -= self.x_hat * sum_dx_hat_xhat
            grad_conv *= self.std_inv / self.M
            self._profile_end("backward", "bn_grad_conv_ms", stage_t0)
        else:
            stage_t0 = self._profile_start()
            grad_conv = grad_bn
            grad_conv *= gamma_data
            grad_conv *= self.std_inv
            self._profile_end("backward", "bn_eval_grad_conv_ms", stage_t0)

        # ----------------------------
        # Conv backward
        # ----------------------------
        stage_t0 = self._profile_start()
        dL_dO_flat = grad_conv.transpose(0, 2, 3, 1).reshape(-1, FN)
        self._profile_end("backward", "grad_flatten_ms", stage_t0)

        stage_t0 = self._profile_start()
        if need_b_grad:
            dL_dB = self.xp.sum(dL_dO_flat, axis=0, keepdims=True)
            if self.b.grad is None:
                self.b.grad = self.xp.zeros_like(self.b.data)
            self.b.grad += dL_dB.reshape(self.b.shape())
        self._profile_end("backward", "bias_grad_ms", stage_t0)

        dL_dW = dL_dCol = None

        stage_t0 = self._profile_start()
        if need_w_grad or need_x_grad:
            dL_dW, dL_dCol = _conv2d_gemm_backward_select(
                col=self.col,
                dO_flat=dL_dO_flat,
                w_data=self.w.data,
                groups=groups,
                FN=FN,
                C_group=C_group,
                KH=KH,
                KW=KW,
                xp=self.xp,
                need_w=need_w_grad,
                need_col=need_x_grad,
            )
        self._profile_end("backward", "conv_backward_gemm_ms", stage_t0)

        stage_t0 = self._profile_start()
        if need_w_grad:
            if self.w.grad is None:
                self.w.grad = self.xp.zeros_like(self.w.data)

            self.w.grad += dL_dW
        self._profile_end("backward", "weight_accum_ms", stage_t0)

        stage_t0 = self._profile_start()
        if need_x_grad:
            if self.x.grad is None:
                self.x.grad = self.xp.zeros_like(self.x.data)

            use_fast_path = (
                self.dilation == (1, 1)
                and self.stride[0] == self.stride[1]
                and self.padding[0] == self.padding[1]
            )

            if use_fast_path:
                self.x.grad += _col2im(
                    dL_dCol,
                    self.x_shape,
                    KH,
                    KW,
                    stride=self.stride[0],
                    padding=self.padding[0],
                    xp=self.xp,
                )
            else:
                self.x.grad += _col2im_ext(
                    dL_dCol,
                    self.x_shape,
                    KH,
                    KW,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    xp=self.xp,
                )
        self._profile_end("backward", "col2im_ms", stage_t0)
    def _get_inputs(self):
        parents = [self.x, self.w]

        if self.b is not None:
            parents.append(self.b)

        parents.append(self.gamma)
        parents.append(self.beta)

        return parents
    
class FusedConvBNAddReLUOp(FusedConvBNReLUOp):
    """
    训练态 Conv2d + BatchNorm2d + Add(identity) + ReLU 融合算子。

    forward:
        conv = Conv2d(x, w, b)
        bn   = BN(conv)
        out  = ReLU(bn + identity)

    backward:
        grad_out -> ReLU backward
                 -> Add backward:
                        identity grad
                        BN grad
                 -> Conv backward
    """

    def forward(
        self,
        x: Tensor,
        identity: Tensor,
        w: Tensor,
        b: Tensor,
        gamma: Tensor,
        beta: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        self.identity = identity

        # 先手动执行 Conv + BN 部分，不能直接调用父类 forward，
        # 因为父类会立刻 ReLU。
        self.x = x
        self.w = w
        self.b = b
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var

        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups
        self.x_shape = x.shape()

        FN, C_group, KH, KW = w.shape()
        N, C_total, H, W_in = self.x_shape

        if groups <= 0 or C_total % groups != 0 or FN % groups != 0:
            raise ValueError("Groups 参数设置非法或输入输出通道数无法被整除")

        if C_group != C_total // groups:
            raise ValueError(
                f"权重输入通道与 groups 不匹配：期望 {C_total // groups}，实际得到 {C_group}"
            )

        use_fast_path = (
            self.dilation == (1, 1)
            and self.stride[0] == self.stride[1]
            and self.padding[0] == self.padding[1]
        )

        stage_t0 = self._profile_start()
        if use_fast_path:
            col, _, _ = _im2col(
                x.data,
                KH,
                KW,
                stride=self.stride[0],
                padding=self.padding[0],
                xp=self.xp,
            )
            out_h = (H + 2 * self.padding[0] - KH) // self.stride[0] + 1
            out_w = (W_in + 2 * self.padding[0] - KW) // self.stride[0] + 1
        else:
            col, _, _, out_h, out_w = _im2col_ext(
                x.data,
                KH,
                KW,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                xp=self.xp,
            )
        self._profile_end("forward", "im2col_ms", stage_t0)

        stage_t0 = self._profile_start()
        conv_flat, self.col = _conv2d_gemm_forward(
            col=col,
            w_data=w.data,
            groups=groups,
            FN=FN,
            C_group=C_group,
            KH=KH,
            KW=KW,
            xp=self.xp
        )
        self._profile_end("forward", "gemm_ms", stage_t0)

        stage_t0 = self._profile_start()
        if b is not None:
            if b.data.size != FN:
                raise ValueError(
                    f"Bias 元素数量必须等于 out_channels={FN}，实际传入 {b.data.size}"
                )
            conv_flat += b.data.reshape(1, FN)
        self._profile_end("forward", "bias_ms", stage_t0)

        stage_t0 = self._profile_start()
        conv_out = conv_flat.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        self._profile_end("forward", "reshape_ms", stage_t0)

        if conv_out.shape != identity.data.shape:
            raise ValueError(
                f"FusedConvBNAddReLUOp 要求 conv_out 和 identity shape 相同，"
                f"conv_out={conv_out.shape}, identity={identity.data.shape}"
            )

        self.out_h = out_h
        self.out_w = out_w
        self.M = N * out_h * out_w

        if self.is_train:
            stage_t0 = self._profile_start()
            mean = self.xp.mean(conv_out, axis=(0, 2, 3), keepdims=True)
            self._profile_end("forward", "bn_mean_ms", stage_t0)

            stage_t0 = self._profile_start()
            x_hat = conv_out
            x_hat -= mean
            var = self.xp.mean(x_hat * x_hat, axis=(0, 2, 3), keepdims=True)
            self._profile_end("forward", "bn_var_ms", stage_t0)

            stage_t0 = self._profile_start()
            unbiased_var = var * (self.M / (self.M - 1)) if self.M > 1 else var

            running_mean.data = (
                (1.0 - self.momentum) * running_mean.data
                + self.momentum * mean
            )

            running_var.data = (
                (1.0 - self.momentum) * running_var.data
                + self.momentum * unbiased_var
            )

            self.mean = mean
            self.var = var
            self._profile_end("forward", "running_stats_ms", stage_t0)
        else:
            stage_t0 = self._profile_start()
            self.mean = running_mean.data
            self.var = running_var.data
            x_hat = conv_out
            x_hat -= self.mean
            self._profile_end("forward", "bn_eval_center_ms", stage_t0)

        stage_t0 = self._profile_start()
        self.std_inv = 1.0 / self.xp.sqrt(self.var + self.eps)
        x_hat *= self.std_inv
        self.x_hat = x_hat

        self.data = self.x_hat * gamma.data
        self.data += beta.data
        self.data += identity.data

        self.relu_mask = self.data > 0
        self.xp.maximum(self.data, 0, out=self.data)
        self._profile_end("forward", "affine_add_relu_ms", stage_t0)

        requires_grad = (
            x.requires_grad
            or identity.requires_grad
            or w.requires_grad
            or (b is not None and b.requires_grad)
            or gamma.requires_grad
            or beta.requires_grad
        )

        self._profile_record_event("forward")
        return Tensor(self.data, requires_grad=requires_grad, creator=self)

    def backward(self, grad=None):
        if grad is not None:
            self.grad = grad

        # ReLU + Add backward
        stage_t0 = self._profile_start()
        grad_add = self.grad * self.relu_mask
        self._profile_end("backward", "relu_mask_ms", stage_t0)

        stage_t0 = self._profile_start()
        if self.identity.requires_grad:
            if self.identity.grad is None:
                self.identity.grad = self.xp.zeros_like(self.identity.data)
            self.identity.grad += grad_add
        self._profile_end("backward", "identity_grad_ms", stage_t0)

        self._backward_bn_conv(grad_add)
        self._profile_record_event("backward")
        return

    def _get_inputs(self):
        parents = [self.x, self.identity, self.w]

        if self.b is not None:
            parents.append(self.b)

        parents.append(self.gamma)
        parents.append(self.beta)

        return parents
