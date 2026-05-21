# mytorch/loss.py

from .tensor import Tensor
from .function import (
    MSE,
    LogSoftmaxOp,
    NLLLossOp,
    FusedMSELossOp,
    FusedCrossEntropyLossOp
)
from .modules import Module


class _Loss(Module):
    """
    损失基类。
    reduction 当前主要保留接口，具体 Function 内部默认 mean。
    """
    reduction: str

    def __init__(self, reduction: str = "mean") -> None:
        super(_Loss, self).__init__()
        self.reduction = reduction


class MSELoss(_Loss):
    def __init__(self, reduction: str = "mean", fused: bool = False) -> None:
        """
        均方误差。

        fused=False:
            使用普通 MSE Op。

        fused=True:
            使用 FusedMSELossOp，CUDA 下走自定义 GPU kernel。
        """
        super().__init__(reduction)

        if reduction != "mean":
            raise NotImplementedError(
                "当前 MSELoss 只支持 reduction='mean'"
            )

        self.fused = fused

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.fused:
            return FusedMSELossOp(reduction=self.reduction)(input, target)

        return MSE()(input, target)


class CrossEntropyLoss(_Loss):
    def __init__(self, reduction: str = "mean", fused: bool = True) -> None:
        super().__init__(reduction)
        self.fused = fused

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: logits, shape = (N, C)
        target: class index, shape = (N,) 或 (N, 1)
        """
        if self.fused:
            return FusedCrossEntropyLossOp()(input, target)

        log_probs = LogSoftmaxOp()(input)
        loss = NLLLossOp()(log_probs, target)
        return loss