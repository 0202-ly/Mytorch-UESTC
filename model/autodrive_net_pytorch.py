import torch
import torch.nn as nn


class AutoDriveNetPyTorch(nn.Module):
    """
    输入: (N, 3, 120, 160)
    输出: (N, 1) -> 预测转向角
    """

    def __init__(self):
        super(AutoDriveNetPyTorch, self).__init__()

        # 卷积层部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # 注意：原代码中 conv5 后没有 ELU
        )

        self.flatten = nn.Flatten()

        # 全连接层部分
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 13, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            # 注意：原代码中 fc3 后没有 ELU
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x