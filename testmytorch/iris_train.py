# train_iris.py
import numpy as np
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- 导入mytorch 框架 ---
from mytorch import (
    Tensor, Module, Linear, ReLU,
    CrossEntropyLoss, Adam, SGD,
    Dataloader, Dataset
)

# 导入自定义的鸢尾花数据集和模型
from dataset.iris_dataset import IrisDataset
from model.iris_model import IrisClassifier


# 如果需要可视化可以取消注释
# from mytorch.utils import make_dot

# 如果需要添加 Graphviz 路径可以取消注释
# graphviz_paths = [
#     r"C:\Program Files\Graphviz\bin",
#     r"C:\Program Files (x86)\Graphviz\bin",
#     r"D:\Program Files\Graphviz\bin",
# ]
#
# for path in graphviz_paths:
#     if os.path.exists(path):
#         os.environ["PATH"] = path + ";" + os.environ.get("PATH", "")
#         print(f"添加 Graphviz 路径: {path}")


# ==========================================================
# 鸢尾花分类模型
# ==========================================================

class IrisModel(Module):
    """
    鸢尾花分类模型
    输入: 4个特征 (花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度)
    输出: 3个类别 (山鸢尾, 杂色鸢尾, 维吉尼亚鸢尾)
    """

    def __init__(self, hidden_size=10):
        super().__init__()
        # 全连接层: 4个输入特征 -> hidden_size个隐藏单元
        self.fc1 = Linear(4, hidden_size)
        self.relu1 = ReLU()

        # 全连接层: hidden_size -> 3个输出类别
        self.fc2 = Linear(hidden_size, 3)

        # 注意: 我们使用 CrossEntropyLoss，所以最后一层不需要激活函数

    def forward(self, x: Tensor) -> Tensor:
        """定义前向传播路径"""
        x = self.fc1(x)  # (batch_size, 4) -> (batch_size, hidden_size)
        x = self.relu1(x)  # ReLU激活
        x = self.fc2(x)  # (batch_size, hidden_size) -> (batch_size, 3)
        return x

    def parameters(self):
        """
        收集所有可学习的参数 (weights 和 biases)
        """
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def predict(self, x_batch: Tensor):
        """预测函数"""
        with self._no_grad():
            outputs = self.forward(x_batch)
            predictions = np.argmax(outputs.data, axis=1)
        return predictions, outputs

    def _no_grad(self):
        """简单的上下文管理器，用于预测时禁用梯度计算"""

        class NoGradContext:
            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        return NoGradContext()


# ==========================================================
#  主训练和评估脚本
# ==========================================================
if __name__ == "__main__":

    # --- 设置 ---
    np.random.seed(42)  # 设置随机种子以确保可重复性
    EPOCHS = 100  # 训练轮数
    BATCH_SIZE = 16  # 批次大小
    LEARNING_RATE = 0.01  # 学习率
    HIDDEN_SIZE = 10  # 隐藏层大小

    print("鸢尾花分类训练")
    print("=" * 50)
    print(f"训练轮数: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"隐藏层大小: {HIDDEN_SIZE}")
    print("=" * 50)

    # --- 加载数据 ---
    try:
        print("\n正在加载鸢尾花数据集...")
        train_dataset = IrisDataset(train=True)
        test_dataset = IrisDataset(train=False)

        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"特征维度: {train_dataset.data.shape[1]}")
        print(f"类别数: {len(np.unique(train_dataset.labels))}")

    except Exception as e:
        print(f"加载数据时出错: {e}")
        print("请确保已正确安装 sklearn: pip install scikit-learn")
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
    print("\n正在初始化模型...")
    model = IrisModel(hidden_size=HIDDEN_SIZE)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # 或者使用 SGD: optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    print("模型初始化完成。")

    # --- 可选: 可视化计算图 ---
    # print("\n正在生成计算图可视化...")
    # 取一个 batch 的数据做一次模拟前向传播
    # visualize_x, visualize_y = next(iter(train_loader))
    # 只需要执行一次 forward 和 loss 计算来构建图
    # v_pred = model.forward(visualize_x)
    # v_loss = loss_fn.forward(v_pred, visualize_y)
    #
    # 调用可视化函数并保存
    # dot = make_dot(v_loss)
    # try:
    #     dot.render('iris_computation_graph', view=False)
    #     print("计算图已保存为 iris_computation_graph.svg")
    # except Exception as e:
    #     print(f"生成计算图失败: {e}")
    #     print("跳过可视化，继续训练...")

    # --- 训练循环 ---
    print(f"\n开始训练 {EPOCHS} 个周期...")
    print("-" * 60)

    # 记录训练历史
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        correct_predictions = 0
        total_samples = 0

        # 训练阶段
        for (x_batch, y_batch) in train_loader:
            optimizer.zero_grad()  # 清除梯度

            # 前向传播
            y_pred = model.forward(x_batch)
            loss = loss_fn.forward(y_pred, y_batch)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算准确率
            predictions = np.argmax(y_pred.data, axis=1)
            labels = y_batch.data
            correct_predictions += np.sum(predictions == labels)
            total_samples += len(labels)

            # 记录损失
            epoch_loss += loss.data
            batch_count += 1

        # 计算平均损失和准确率
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        # 测试阶段
        test_correct = 0
        test_total = 0

        for (x_batch, y_batch) in test_loader:
            y_pred = model.forward(x_batch)
            predictions = np.argmax(y_pred.data, axis=1)
            labels = y_batch.data

            test_correct += np.sum(predictions == labels)
            test_total += len(labels)

        test_accuracy = test_correct / test_total if test_total > 0 else 0
        test_accuracies.append(test_accuracy)

        end_time = time.time()
        epoch_time = end_time - start_time

        # 每10个epoch打印一次进度，或者第一个和最后一个epoch也打印
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch + 1:3d}/{EPOCHS} - "
                  f"耗时: {epoch_time:.2f}s - "
                  f"损失: {avg_loss:.4f} - "
                  f"训练准确率: {train_accuracy * 100:.2f}% - "
                  f"测试准确率: {test_accuracy * 100:.2f}%")

    print("-" * 60)
    print("训练完成。")

    # --- 最终评估 ---
    print("\n正在测试集上进行最终评估...")
    final_test_correct = 0
    final_test_total = 0
    all_predictions = []
    all_labels = []

    for (x_batch, y_batch) in test_loader:
        y_pred = model.forward(x_batch)
        predictions = np.argmax(y_pred.data, axis=1)
        labels = y_batch.data

        final_test_correct += np.sum(predictions == labels)
        final_test_total += len(labels)

        all_predictions.append(predictions)
        all_labels.append(labels)

    # 合并所有批次的预测结果
    all_predictions_np = np.concatenate(all_predictions)
    all_labels_np = np.concatenate(all_labels)

    final_accuracy = final_test_correct / final_test_total

    print("-" * 30)
    print(f"最终测试集准确率: {final_accuracy * 100:.2f}%")
    print(f"正确数/总数: {final_test_correct}/{final_test_total}")
    print("-" * 30)

    # 打印混淆矩阵（简化版）
    print("\n混淆矩阵（行: 真实标签, 列: 预测标签）:")
    class_names = ['山鸢尾', '杂色鸢尾', '维吉尼亚鸢尾']

    # 创建3x3的混淆矩阵
    confusion_matrix = np.zeros((3, 3), dtype=int)
    for true_label, pred_label in zip(all_labels_np, all_predictions_np):
        confusion_matrix[true_label, pred_label] += 1

    # 打印混淆矩阵
    print("     预测:  0     1     2")
    print("     " + "-" * 23)
    for i in range(3):
        row_str = f"真实 {i}: "
        for j in range(3):
            row_str += f"{confusion_matrix[i, j]:4d}  "
        print(row_str + f" ({class_names[i]})")

    print("-" * 30)

    # --- 交互式预测示例 ---
    print("\n交互式预测示例:")
    print("鸢尾花特征: [花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]")

    # 一些测试样本
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # 山鸢尾
        [6.7, 3.0, 5.2, 2.3],  # 维吉尼亚鸢尾
        [5.9, 3.0, 4.2, 1.5],  # 杂色鸢尾
    ]

    sample_names = ['山鸢尾 (Iris-setosa)',
                    '维吉尼亚鸢尾 (Iris-virginica)',
                    '杂色鸢尾 (Iris-versicolor)']

    for i, sample in enumerate(test_samples):
        # 将样本转换为Tensor
        sample_tensor = Tensor(np.array(sample, dtype=np.float32).reshape(1, -1), requires_grad=False)

        # 预测
        predictions, outputs = model.predict(sample_tensor)

        predicted_class = predictions[0]

        # 获取概率（softmax）
        output_data = outputs.data[0]
        exp_outputs = np.exp(output_data - np.max(output_data))  # 数值稳定
        probabilities = exp_outputs / np.sum(exp_outputs)

        print(f"\n样本 {i + 1}: {sample}")
        print(f"  真实类别: {sample_names[i]}")
        print(f"  预测类别: {class_names[predicted_class]}")
        print(f"  类别概率:")
        for j in range(3):
            print(f"    {class_names[j]}: {probabilities[j]:.4f}")

    print("\n" + "=" * 50)
    print("鸢尾花分类训练完成!")
    print("=" * 50)

    # --- 可选: 可视化训练曲线 ---
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.show()

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
        plt.plot(test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        plt.tight_layout()
        plt.savefig('iris_training_results.png', dpi=300)
        print(f"\n训练曲线已保存为 'iris_training_results.png'")

    except ImportError:
        print("\n提示: 安装 matplotlib 可以可视化训练曲线")
        print("命令: pip install matplotlib")