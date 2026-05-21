from mytorch import Module, Conv2d, Linear, ELU, Flatten
from mytorch import Tensor


class AutoDriveNet(Module):
    """
    输入: (N, 3, 120, 160)
    输出: (N, 1) -> 预测转向角
    """

    def __init__(self):
        super().__init__()
        # ... (此处粘贴您原有的 AutoDriveNet __init__ 代码) ...
        # 注意：如果您实现了 Dropout，记得在这里加上

        self.conv1 = Conv2d(3, 24, kernel_size=5, stride=2)
        self.elu1 = ELU()
        self.conv2 = Conv2d(24, 36, kernel_size=5, stride=2)
        self.elu2 = ELU()
        self.conv3 = Conv2d(36, 48, kernel_size=5, stride=2)
        self.elu3 = ELU()
        self.conv4 = Conv2d(48, 64, kernel_size=3, stride=1)
        self.elu4 = ELU()
        self.conv5 = Conv2d(64, 64, kernel_size=3, stride=1)

        self.flatten = Flatten()

        self.fc1 = Linear(64 * 8 * 13, 100)
        self.elu5 = ELU()
        self.fc2 = Linear(100, 50)
        self.elu6 = ELU()
        self.fc3 = Linear(50, 10)
        self.fc4 = Linear(10, 1)

    def forward(self, x: Tensor) -> Tensor:
        # ... (此处粘贴您原有的 forward 代码) ...
        x = self.elu1(self.conv1(x))
        x = self.elu2(self.conv2(x))
        x = self.elu3(self.conv3(x))
        x = self.elu4(self.conv4(x))
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.elu5(self.fc1(x))
        x = self.elu6(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x
