import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- 导入 mytorch 组件 ---
from mytorch.loss import MSELoss
from mytorch.optim import Adam
from mytorch.dataloader import Dataloader
from mytorch.dataset import AutoDriveDataset

from .model.autodrive_net import AutoDriveNet
os.environ['NVCC_PREPEND_FLAGS'] = '-allow-unsupported-compiler'
# --- 检查 GPU 是否可用 ---
try:
    import cupy

    DEVICE = 'cuda'
    print(f"检测到 CuPy，将使用 GPU ({DEVICE}) 进行训练。")
except ImportError:
    DEVICE = 'cpu'
    print(f"未检测到 CuPy，将使用 CPU ({DEVICE}) 进行训练。")


# =========================================================================
# 1. 辅助工具：早停类 (EarlyStopping)
# =========================================================================
class EarlyStopping:
    """
    早停机制：当验证集 Loss 在 patience 个 epoch 内没有提升时，停止训练。
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: 容忍多少个 epoch 没有提升
        :param min_delta: 只有 loss 降低超过此值才算提升 (防止微小的抖动)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0  # 当前连续未提升计数
        self.best_loss = None  # 历史最佳 Loss
        self.early_stop = False  # 是否触发停止

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # 当前 Loss 没有明显比历史最佳 Loss 低 -> 计数 +1
            self.counter += 1
            print(f"   [EarlyStopping] 计数: {self.counter}/{self.patience} (最佳: {self.best_loss:.5f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 发现更优 Loss -> 重置计数，更新最佳值
            self.best_loss = val_loss
            self.counter = 0


# =========================================================================
# 2. 辅助工具：保存模型与验证函数
# =========================================================================
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 注意：如果参数在 GPU 上，pickle dump 前最好转回 CPU 数据，
    # 但我们这里直接保存 data (numpy/cupy array)。
    # 为了通用性，建议保存时确保转为 numpy
    params_data = []
    for p in model.parameters():
        if hasattr(p.data, 'get'):  # 如果是 cupy array
            params_data.append(p.data.get())
        else:
            params_data.append(p.data)

    with open(path, 'wb') as f:
        pickle.dump(params_data, f)


def validate(model, val_loader, criterion, device='cpu'):
    """验证循环：计算验证集平均 Loss"""
    model.eval()  # 切换到评估模式 (关闭 Dropout)
    val_loss_sum = 0
    num_batches = 0

    # 验证不需要计算梯度，纯前向传播
    for imgs, labels in val_loader:
        # --- 数据迁移至 GPU ---
        if device == 'cuda':
            imgs = imgs.cuda()
            labels = labels.cuda()

        pre_labels = model(imgs)
        loss = criterion(pre_labels, labels)

        # 将 loss 数据转为 python float (会自动处理 cupy -> float)
        val_loss_sum += float(loss.data)
        num_batches += 1

    if num_batches == 0:
        return 0.0
    return val_loss_sum / num_batches


# =========================================================================
# 3. 主程序
# =========================================================================
if __name__ == "__main__":
    # --- A. 参数配置 ---
    batch_size = 64
    max_epochs = 100
    lr = 1e-4
    patience = 5  # 早停耐心值

    # --- B. 准备数据与模型 ---
    print("正在加载数据与模型...")
    model = AutoDriveNet()

    # --- 模型迁移至 GPU ---
    if DEVICE == 'cuda':
        model.cuda()  # 递归将所有参数移动到显存

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = MSELoss()

    # 训练集
    train_dataset = AutoDriveDataset(mode="train", data_root="./")
    train_loader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)

    # 验证集
    val_dataset = AutoDriveDataset(mode="val", data_root="./")
    val_loader = Dataloader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=None)

    # 初始化早停对象
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)

    train_history = []
    val_history = []

    # 用于手动保存最佳模型 (配合早停使用)
    best_val_loss = float('inf')

    print(f"开始训练 (Device: {DEVICE}, Max Epochs: {max_epochs}, Patience: {patience})...")

    # --- C. 训练循环 ---
    for epoch in range(1, max_epochs + 1):
        # 1. 训练阶段
        model.train()
        train_loss_sum = 0
        train_batches = 0

        for imgs, labels in train_loader:
            # --- 数据迁移至 GPU ---
            if DEVICE == 'cuda':
                imgs = imgs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            pre_labels = model(imgs)
            loss = criterion(pre_labels, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.data)
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0
        train_history.append(avg_train_loss)

        # 2. 验证阶段
        avg_val_loss = validate(model, val_loader, criterion, device=DEVICE)
        val_history.append(avg_val_loss)

        # 3. 打印进度
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.5f} | Val Loss = {avg_val_loss:.5f}")

        # 4. 保存最佳模型逻辑
        if avg_val_loss < best_val_loss:
            print(f"   >>> 发现新纪录 (Loss: {best_val_loss:.5f} -> {avg_val_loss:.5f})，保存模型...")
            best_val_loss = avg_val_loss
            save_model(model, "results/best_model.pkl")

        # 5. 检查早停
        early_stopping(avg_val_loss)  # 传入当前验证集 Loss
        if early_stopping.early_stop:
            print(f"\n[停止训练] 验证集 Loss 连续 {patience} 轮未下降，触发早停机制。")
            print(f"最佳模型已保存在 results/best_model.pkl (Loss: {early_stopping.best_loss:.5f})")
            break

    # --- D. 绘制曲线 ---
    plt.figure(figsize=(10, 5))
    epochs_ran = range(1, len(train_history) + 1)
    plt.plot(epochs_ran, train_history, label='Train Loss')
    plt.plot(epochs_ran, val_history, label='Val Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training & Validation Loss (Device: {DEVICE})')
    plt.legend()
    plt.grid(True)
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/loss_curve.png")
    print("训练结束。Loss 曲线已保存。")