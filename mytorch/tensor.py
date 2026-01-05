from typing import Union,List
import numpy as np

NdArray = Union['np.ndarray','cuda.ndarray']
Arrayble = Union[NdArray, List]

class Tensor:
    def __init__(self,
                 data: Arrayble,
                 device = "cpu",
                 requires_grad = False,
                 creator=None,
                    ):
        if isinstance(data, List):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            data = np.asarray(data)
        self.grad = None
        self._device = device
        self.creator = creator
        self.index = -1
        self.requires_grad = requires_grad
        self.data = data

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
        self.grad = np.zeros_like(self.data)

    @staticmethod
    def random_matrix(*shape, requires_grad=False):
        data = np.random.randn(*shape) * 0.01
        return Tensor(data, requires_grad=requires_grad)

    def add_(self, other, alpha=1):
        # 实现 self.data = self.data + (alpha * other.data)
        if isinstance(other, Tensor):
            np.add(self.data, other.data * alpha, out=self.data)
        else:
            np.add(self.data, other * alpha, out=self.data)
        return self

    def mul_(self, value):
        # 实现 self.data = self.data * value
        np.multiply(self.data, value, out=self.data)
        return self

    def addcmul_(self, tensor1, tensor2, value=1):
        # --- 修正 ---：检查输入是 Tensor 还是 np.ndarray
        t1_data = tensor1.data if isinstance(tensor1, Tensor) else tensor1
        t2_data = tensor2.data if isinstance(tensor2, Tensor) else tensor2

        # 实现 self.data = self.data + (value * (t1_data * t2_data))
        np.add(self.data, value * np.multiply(t1_data, t2_data), out=self.data)
        return self

    def addcdiv_(self, tensor_num, tensor_den, value=1):
        # --- 修正 ---：检查输入是 Tensor 还是 np.ndarray
        # (Adam 没触发这个 bug, 但 Adagrad 会触发)
        num_data = tensor_num.data if isinstance(tensor_num, Tensor) else tensor_num
        den_data = tensor_den.data if isinstance(tensor_den, Tensor) else tensor_den

        # 实现 self.data = self.data + (value * (num_data / den_data))
        np.add(self.data, value * np.divide(num_data, den_data), out=self.data)
        return self

    def sqrt(self):
        # 返回一个新的 Tensor，包含 sqrt(self.data)
        # 注意：Adagrad 的 std=sum.sqrt() 不是原地操作
        return Tensor(np.sqrt(self.data))

    @staticmethod
    def full_like(other_tensor, fill_value):
        # 辅助函数，用于创建与另一个 Tensor 形状相同且填满特定值的 Tensor
        return Tensor(np.full_like(other_tensor.data, fill_value), requires_grad=False)
    def forward(self):
        pass

    def backward(self):
        """
        执行整个计算图的反向传播。
        从这个 Tensor (通常是 loss) 开始。
        """
        from .function import Function
        if self.creator is None:
            # 这是根节点（例如输入数据），不需要反向传播
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
                # Tensor: 它的 parent 是创建它的 Op
                if v.creator is not None:
                    parents = [v.creator]
#----修改：------
            elif isinstance(v, Function):
                # Ops with 2 inputs (a, b)
                parents =  v._get_inputs()

            for parent in parents:
                if parent is not None:
                    build_topo(parent)
            topo.append(v)

# ---------------------------------
        build_topo(self)

        # 2. 初始化所有节点的梯度
        for node in visited:
            if isinstance(node, Tensor) and node.requires_grad:
                # Tensors 的梯度在 zero_grad() 中初始化
                # 这里确保它们存在
                if node.grad is None:
                    node.grad = np.zeros_like(node.data, dtype=np.float64)
            elif not isinstance(node, Tensor):  # 是 Op
                # Op 的梯度是其中间结果的梯度
                if node.data is not None:
                    node.grad = np.zeros_like(node.data, dtype=np.float64)

        # 3. 设置最终节点（损失节点）的梯度为 1
        # --- 修正：self 就是最终的 loss Tensor ---
        self.grad = np.array(1.0).reshape(self.data.shape)

        # 4. 按拓扑排序的【逆序】执行每个节点的 backward 方法
        for v in reversed(topo):
            if isinstance(v, Tensor):
                # Tensor: 将梯度传递给它的创建者 (Op)
                if v.creator is not None and v.grad is not None:
                    if v.creator.grad is None:
                        v.creator.grad = np.zeros_like(v.creator.data)
                    # --- 修正：梯度是累加的 ---
                    v.creator.grad += v.grad
            else:
                # Op: 调用自己的 backward() 来计算局部梯度并分发
                if v.grad is not None:  # 确保接收到了梯度
                    v.backward()

