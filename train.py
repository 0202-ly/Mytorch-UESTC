import numpy as np
import time
from urllib.error import URLError  # 导入错误类型以便捕获

# --- 导入mytorch 框架 ---
from mytorch import (
    Tensor, Module, Linear, Conv2d, MaxPool,
    Flatten, ReLU, CrossEntropyLoss, Adam,
    Dataloader,
    Dataset,
    MnistDataset
)


class LeNet(Module):
    def __init__(self):
        """
        定义模型架构，使用您在 modules.py 中定义的所有层。
        """
        super().__init__()
        # (N, 1, 28, 28) -> (N, 6, 28, 28)
        self.conv1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)  #
        self.relu1 = ReLU()
        # (N, 6, 28, 28) -> (N, 6, 14, 14)
        self.pool1 = MaxPool(kernel_size=2, stride=2)  #

        # (N, 6, 14, 14) -> (N, 16, 10, 10)
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)  #
        self.relu2 = ReLU()
        # (N, 16, 10, 10) -> (N, 16, 5, 5)
        self.pool2 = MaxPool(kernel_size=2, stride=2)  #

        # (N, 16, 5, 5) -> (N, 16*5*5) = (N, 400)
        self.flatten = Flatten()  #

        # (N, 400) -> (N, 120)
        self.fc1 = Linear(16 * 5 * 5, 120)  #
        self.relu3 = ReLU()

        # (N, 120) -> (N, 84)
        self.fc2 = Linear(120, 84)  #
        self.relu4 = ReLU()

        # (N, 84) -> (N, 10) (10 个类别)
        self.fc3 = Linear(84, 10)  #

    def forward(self, x: Tensor) -> Tensor:
        """定义前向传播路径"""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

    def parameters(self):
        """
        收集所有可学习的参数 (weights 和 biases)
        """
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params


# ==========================================================
# 数据加载和预处理
# ==========================================================


# ==========================================================
#  主训练和评估脚本
# ==========================================================
if __name__ == "__main__":

    # --- 设置 ---
    np.random.seed(42)  #
    EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    # --- 加载数据  ---
    try:

        train_dataset = MnistDataset(data_root="data", train=True, download=True)
        test_dataset = MnistDataset(data_root="data", train=False, download=True)

    except (FileNotFoundError, RuntimeError, URLError) as e:
        print(f"加载或下载数据时出错: {e}")
        print("请检查您的网络连接或 'data' 文件夹 的权限。")
        exit()

    train_loader = Dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=None
    )

    test_loader = Dataloader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=None
    )

    # --- 初始化模型、损失和优化器 ---
    print("正在初始化模型...")
    model = LeNet()
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    print("模型初始化完成。")

    # --- 训练循环 ---
    print(f"开始训练 {EPOCHS} 个周期...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0

        for (x_batch, y_batch) in train_loader:  #
            optimizer.zero_grad()  #
            y_pred = model.forward(x_batch)  #
            loss = loss_fn.forward(y_pred, y_batch)  #
            loss.backward()  #
            optimizer.step()  #
            epoch_loss += loss.data
            batch_count += 1

        end_time = time.time()
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch + 1}/{EPOCHS} - "
              f"耗时: {end_time - start_time:.2f}s - "
              f"平均损失 : {avg_loss:.6f}")

    print("训练完成。")

    # --- 评估  ---
    print("正在测试集上评估...")
    all_preds = []
    all_labels = []
    for (x_batch, y_batch) in test_loader:  #
        y_pred_tensor = model.forward(x_batch)  #
        y_pred_labels = np.argmax(y_pred_tensor.data, axis=1)  #
        all_preds.append(y_pred_labels)
        all_labels.append(y_batch.data)

    all_preds_np = np.concatenate(all_preds)  #
    all_labels_np = np.concatenate(all_labels)
    accuracy = np.mean(all_preds_np == all_labels_np)  #

    print("-" * 30)
    print(f"测试集准确率: {accuracy * 100:.2f}%")
    print("-" * 30)