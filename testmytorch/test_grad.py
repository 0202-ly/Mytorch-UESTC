import numpy as np
from mytorch import (
    Tensor, Module, Linear, Conv2d, AvgPool,
    Flatten, Sigmoid, MSELoss
)
from mytorch.utils import grad_check_model


class GradCheckNet(Module):
    """
    专为梯度检验设计的平滑模型
    结构：Conv -> Sigmoid -> AvgPool -> Flatten -> Linear -> Sigmoid -> Linear
    """

    def __init__(self):
        super().__init__()
        # 输入：1通道，8x8图像
        self.conv1 = Conv2d(1, 4, kernel_size=3, padding=1)
        self.sigmoid = Sigmoid()
        self.pool1 = AvgPool(kernel_size=2)  # 变为 4x4
        self.flatten = Flatten()
        self.fc1 = Linear(4 * 4 * 4, 10)
        self.fc2 = Linear(10, 2)  # 输出 2 类

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


def run_precision_grad_check():
    # 1. 构造测试数据
    # 强制使用 float64 提高数值微分精度
    x_data = np.random.randn(1, 1, 8, 8).astype(np.float64)
    y_data = np.random.randn(1, 2).astype(np.float64)

    inputs = Tensor(x_data, requires_grad=False)
    targets = Tensor(y_data, requires_grad=False)

    # 2. 初始化模型
    model = GradCheckNet()

    # 3. 关键步骤：将模型所有参数提升到 float64 精度
    for p in model.parameters():
        p.data = p.data.astype(np.float64)

    # 4. 使用平滑的损失函数
    loss_fn = MSELoss()

    # 5. 执行检验 (减小 eps 到 1e-7)
    # 理想情况下，误差应下降到 1e-7 ~ 1e-9 之间
    grad_check_model(model, loss_fn, inputs, targets, eps=1e-7)


if __name__ == "__main__":
    run_precision_grad_check()