from typing import Union, List
import numpy as np

# ---尝试导入 cupy ---
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

NdArray = Union['np.ndarray', 'cp.ndarray']
Arrayble = Union[NdArray, List]


class Tensor:
    def __init__(self,
                 data: Arrayble,
                 device: str = None,  # 修改：允许初始化指定 device
                 requires_grad: bool = False,
                 creator=None,
                 ):

        # 1. 自动推断 device 或处理 list 输入
        if isinstance(data, List):
            data = np.array(data)

        # 如果没有指定 device，根据传入数据的类型判断
        if device is None:
            if GPU_AVAILABLE and isinstance(data, cp.ndarray):
                device = 'cuda'
            else:
                device = 'cpu'

        self._device = device

        # 2. 根据 device 转换数据类型
        if self._device == 'cuda' and GPU_AVAILABLE:
            if not isinstance(data, cp.ndarray):
                data = cp.asarray(data)
        else:
            # CPU 模式
            if GPU_AVAILABLE and isinstance(data, cp.ndarray):
                data = data.get()  # 从 GPU 取回
            elif not isinstance(data, np.ndarray):
                data = np.asarray(data)

        self.grad = None
        self.creator = creator
        self.index = -1
        self.requires_grad = requires_grad
        self.data = data

    @property
    def xp(self):
        """返回当前 Tensor 使用的后端库 (numpy 或 cupy)"""
        if self._device == 'cuda' and GPU_AVAILABLE:
            return cp
        return np

    def cuda(self):
        """将 Tensor 移动到 GPU"""
        if self._device == 'cuda':
            return self
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not installed or GPU not available.")

        # 移动数据和梯度
        self.data = cp.asarray(self.data)
        if self.grad is not None:
            self.grad = cp.asarray(self.grad)

        self._device = 'cuda'
        return self

    def cpu(self):
        """将 Tensor 移动到 CPU"""
        if self._device == 'cpu':
            return self

        # 移动数据和梯度
        if GPU_AVAILABLE and isinstance(self.data, cp.ndarray):
            self.data = self.data.get()
        if self.grad is not None and GPU_AVAILABLE and isinstance(self.grad, cp.ndarray):
            self.grad = self.grad.get()

        self._device = 'cpu'
        return self

    def ndim(self):
        return self.data.ndim

    def shape(self):
        return self.data.shape

    def get_grad(self):
        return self.grad

    def dtype(self):
        return self.data.dtype

    def device(self):
        return self._device

    def __repr__(self):
        return str(self)

    def __str__(self):
        # 打印时必须确保数据在 CPU 上
        data_to_print = self.data
        if self._device == 'cuda' and GPU_AVAILABLE:
            data_to_print = self.data.get()
        return f"Tensor({data_to_print}, device='{self._device}', requires_grad={self.requires_grad})"

    # --- 静态方法需要返回 CPU Tensor 默认，或者根据上下文优化 ---
    @staticmethod
    def eye(*shape):
        return Tensor(np.eye(*shape))

    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(*shape))

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(*shape))

    def zero_grad(self):
        # 使用 self.xp 确保创建的 0 梯度与 data 在同一设备
        self.grad = self.xp.zeros_like(self.data)

    @staticmethod
    def random_matrix(*shape, requires_grad=False):
        data = np.random.randn(*shape) * 0.01
        return Tensor(data, requires_grad=requires_grad)

    # --- 修正原地操作：使用 self.xp 而不是 np ---
    def add_(self, other, alpha=1):
        # 实现 self.data = self.data + (alpha * other.data)
        if isinstance(other, Tensor):
            # 注意：other 也必须在同一设备，否则 cupy 会报错
            self.xp.add(self.data, other.data * alpha, out=self.data)
        else:
            self.xp.add(self.data, other * alpha, out=self.data)
        return self

    def mul_(self, value):
        self.xp.multiply(self.data, value, out=self.data)
        return self

    def addcmul_(self, tensor1, tensor2, value=1):
        t1_data = tensor1.data if isinstance(tensor1, Tensor) else tensor1
        t2_data = tensor2.data if isinstance(tensor2, Tensor) else tensor2

        # 使用 self.xp
        self.xp.add(self.data, value * self.xp.multiply(t1_data, t2_data), out=self.data)
        return self

    def addcdiv_(self, tensor_num, tensor_den, value=1):
        num_data = tensor_num.data if isinstance(tensor_num, Tensor) else tensor_num
        den_data = tensor_den.data if isinstance(tensor_den, Tensor) else tensor_den

        # 使用 self.xp
        self.xp.add(self.data, value * self.xp.divide(num_data, den_data), out=self.data)
        return self

    def sqrt(self):
        # 返回一个新的 Tensor，包含 sqrt(self.data)
        # 注意：这里需要传递 device 信息，保持与 self 一致
        return Tensor(self.xp.sqrt(self.data), device=self._device, requires_grad=self.requires_grad)

    @staticmethod
    def full_like(other_tensor, fill_value):
        # 修正：根据 other_tensor 的 xp 创建数据，并继承 device
        data = other_tensor.xp.full_like(other_tensor.data, fill_value)
        return Tensor(data, device=other_tensor.device(), requires_grad=False)

    def forward(self):
        pass

    def backward(self):
        """
        执行整个计算图的反向传播。
        """
        from .function import Function
        if self.creator is None:
            return

        if self.data is None:
            raise ValueError("在执行反向传播前，请先执行前向传播。")

        # 1. 使用拓扑排序构建从前到后的节点列表
        topo = []
        visited = set()

        def build_topo(v):
            if v in visited:
                return
            visited.add(v)
            parents = []
            if isinstance(v, Tensor):
                if v.creator is not None:
                    parents = [v.creator]
            elif isinstance(v, Function):
                parents = v._get_inputs()

            for parent in parents:
                if parent is not None:
                    build_topo(parent)
            topo.append(v)

        build_topo(self)

        # 2. 初始化所有节点的梯度
        for node in visited:
            if isinstance(node, Tensor) and node.requires_grad:
                if node.grad is None:
                    # 使用 node.xp 确保梯度在正确的设备上
                    node.grad = node.xp.zeros_like(node.data, dtype=np.float64)
            elif not isinstance(node, Tensor):
                if node.data is not None:
                    # Op 的梯度是其中间结果的梯度
                    node.grad = node.xp.zeros_like(node.data, dtype=np.float64)

        # 3. 设置最终节点（损失节点）的梯度为 1
        # 修正：创建位于正确设备上的 1.0
        self.grad = self.xp.array(1.0).reshape(self.data.shape)

        # 4. 按拓扑排序的【逆序】执行每个节点的 backward 方法
        for v in reversed(topo):
            if isinstance(v, Tensor):
                if v.creator is not None and v.grad is not None:
                    if v.creator.grad is None:
                        # 确保创建者梯度设备正确
                        v.creator.grad = v.creator.xp.zeros_like(v.creator.data)
                    v.creator.grad += v.grad
            else:
                if v.grad is not None:
                    v.backward()