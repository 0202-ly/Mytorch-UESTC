from mytorch import Module, Conv2d, Linear, ReLU,ELU, MaxPool, Flatten, Tensor


# ===============================================================
#  模型 1: LeNet (用于 MNIST 分类任务)
# ===============================================================
class LeNet(Module):
    def __init__(self):
        super().__init__()
        # input: (N, 1, 28, 28)
        # Conv1: (N, 6, 28, 28) -> MaxPool: (N, 6, 14, 14)
        self.conv1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.relu1 = ReLU()
        self.pool1 = MaxPool(kernel_size=2, stride=2)

        # Conv2: (N, 16, 10, 10) -> MaxPool: (N, 16, 5, 5)
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.relu2 = ReLU()
        self.pool2 = MaxPool(kernel_size=2, stride=2)

        self.flatten = Flatten()

        # Linear: 16 * 5 * 5 = 400
        self.fc1 = Linear(400, 120)
        self.relu3 = ReLU()
        self.fc2 = Linear(120, 84)
        self.relu4 = ReLU()
        self.fc3 = Linear(84, 10)  # 10分类

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

    def parameters(self):
        """收集参数用于优化器"""
        return (self.conv1.parameters() +
                self.conv2.parameters() +
                self.fc1.parameters() +
                self.fc2.parameters() +
                self.fc3.parameters())


# ===============================================================
#  模型 2: DonkeyNet (用于自动驾驶回归任务)
# ===============================================================
class AutoDriveNet(Module):
    """
    仿照 Paddle 版 AutoDriveNet 实现的自动驾驶模型
    输入: (N, 3, 120, 160)
    输出: (N, 1) -> 预测转向角
    """

    def __init__(self):
        super().__init__()

        # --- 卷积特征提取模块 (Conv Layers) ---

        # 1. Conv2D(3, 24, 5, stride=2)
        # H_out = (120 - 5)//2 + 1 = 58
        # W_out = (160 - 5)//2 + 1 = 78
        self.conv1 = Conv2d(3, 24, kernel_size=5, stride=2)
        self.elu1 = ELU()

        # 2. Conv2D(24, 36, 5, stride=2)
        # H_out = (58 - 5)//2 + 1 = 27
        # W_out = (78 - 5)//2 + 1 = 37
        self.conv2 = Conv2d(24, 36, kernel_size=5, stride=2)
        self.elu2 = ELU()

        # 3. Conv2D(36, 48, 5, stride=2)
        # H_out = (27 - 5)//2 + 1 = 12
        # W_out = (37 - 5)//2 + 1 = 17
        self.conv3 = Conv2d(36, 48, kernel_size=5, stride=2)
        self.elu3 = ELU()

        # 4. Conv2D(48, 64, 3) -> 默认 stride=1
        # H_out = (12 - 3)//1 + 1 = 10
        # W_out = (17 - 3)//1 + 1 = 15
        self.conv4 = Conv2d(48, 64, kernel_size=3, stride=1)
        self.elu4 = ELU()

        # 5. Conv2D(64, 64, 3) -> 默认 stride=1
        # H_out = (10 - 3)//1 + 1 = 8
        # W_out = (15 - 3)//1 + 1 = 13
        # 最终特征图大小: (64, 8, 13)
        self.conv5 = Conv2d(64, 64, kernel_size=3, stride=1)
        # 原代码这里有 Dropout(0.5)，mytorch 暂未实现，故跳过

        # --- 展平层 ---
        self.flatten = Flatten()

        # --- 线性变换模块 (Linear Layers) ---

        # 输入维度: 64 * 8 * 13 = 6656
        self.fc1 = Linear(64 * 8 * 13, 100)
        self.elu5 = ELU()

        self.fc2 = Linear(100, 50)
        self.elu6 = ELU()

        # Paddle代码: Linear(50, 10) -> Linear(10, 1)
        self.fc3 = Linear(50, 10)
        # 原代码这里没有激活函数，直接接下一层

        self.fc4 = Linear(10, 1)  # 输出 1 个值 (Steering)

    def forward(self, x: Tensor) -> Tensor:
        """前向推理"""
        # 注意：AutoDriveDataset 应该已经输出了 (N, 3, 120, 160) 的 Tensor

        # --- Conv Layers ---
        x = self.elu1(self.conv1(x))
        x = self.elu2(self.conv2(x))
        x = self.elu3(self.conv3(x))
        x = self.elu4(self.conv4(x))
        x = self.conv5(x)
        # (无 dropout)

        # --- Flatten ---
        x = self.flatten(x)

        # --- Linear Layers ---
        x = self.elu5(self.fc1(x))
        x = self.elu6(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)

        return x

    def parameters(self):
        """收集所有层的参数"""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.conv3.parameters())
        params.extend(self.conv4.parameters())
        params.extend(self.conv5.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        params.extend(self.fc4.parameters())
        return params