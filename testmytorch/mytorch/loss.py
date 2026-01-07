# --- 修正：导入我们新定义的 MSE Op ---
from .tensor import Tensor
from .function import MSE,LogSoftmaxOp, NLLLossOp
from .modules import Module # --- 修正：loss 继承自 Module (来自 tensor.py) ---


class _Loss(Module):
    '''
    损失的基类
    '''
    reduction: str  # none | mean | sum

    def __init__(self, reduction: str = "mean") -> None:
        super(_Loss, self).__init__()
        self.reduction = reduction

# ... squared_loss (这是一个辅助函数，不动) ...

class MSELoss(_Loss):
    def __init__(self, reduction: str = "mean") -> None:
        '''
        均方误差
        '''
        super().__init__(reduction)
        # --- 修正：创建一个 Op 实例 ---
        # 注意：我们的 MSE Op 内部硬编码了 "mean"
        self.op = MSE()


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # --- 修正：使用 MSE Op 来构建图 ---
        return self.op(input, target)


class CrossEntropyLoss(_Loss):
    def __init__(self, reduction: str = "mean") -> None:
        """
        此模块将 LogSoftmax 和 NLLLoss 组合在一个类中。
        """
        super().__init__(reduction)
        # (注意：我们的 NLLLossOp 内部硬编码了 "mean")
        self.log_softmax_op = LogSoftmaxOp()
        self.nll_loss_op = NLLLossOp()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: 模型的原始输出 (Logits)，形状 (N, C)
        target: 类别索引 (整数 0 到 C-1)，形状 (N,)
        """
        # 1. 计算 LogSoftmax
        log_probs = self.log_softmax_op(input)

        # 2. 计算 NLLLoss
        loss = self.nll_loss_op(log_probs, target)

        return loss
