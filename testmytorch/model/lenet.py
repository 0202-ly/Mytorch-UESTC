# mytorch/models/lenet.py
from sys import modules

from mytorch import Module, Conv2d, Linear, ReLU, MaxPool, Flatten
from mytorch import Tensor

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
        self.fc3 = Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

    def parameters(self):
        return (self.conv1.parameters() +
                self.conv2.parameters() +
                self.fc1.parameters() +
                self.fc2.parameters() +
                self.fc3.parameters())