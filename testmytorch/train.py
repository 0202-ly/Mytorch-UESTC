import numpy as np
import time
from urllib.error import URLError  # 导入错误类型以便捕获

# --- 导入mytorch 框架 ---
from mytorch import (
    Tensor, Module, Linear, Conv2d, MaxPool,
    Flatten, ReLU, CrossEntropyLoss, Adam,
    Dataloader,
    Dataset,
    make_dot
)
from dataset.mnist_dataset import MnistDataset
from model.lenet import  LeNet


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

    # ---可视化计算图 ---
    print("正在生成计算图可视化...")
    # 取一个 batch 的数据做一次模拟前向传播
    visualize_x, visualize_y = next(iter(train_loader))
    # 只需要执行一次 forward 和 loss 计算来构建图
    v_pred = model.forward(visualize_x)
    v_loss = loss_fn.forward(v_pred, visualize_y)

    # 调用可视化函数并保存
    # 注意：确保你已经按照之前的建议将 make_dot 放在了 mytorch/__init__.py 或相关路径
    dot = make_dot(v_loss)
    dot.render('lenet_computation_graph', view=True)
    print("计算图已保存为 lenet_computation_graph.svg")


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