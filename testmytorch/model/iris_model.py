# mytorch/iris_model.py
import numpy as np
from mytorch import Module, Linear, ReLU, Sigmoid
# Softmax
from mytorch import Tensor


class IrisSimpleNN(Module):
    """简单的全连接神经网络用于鸢尾花分类"""

    def __init__(self, input_size=4, hidden_size=16, output_size=3):
        super().__init__()

        # 网络结构
        self.fc1 = Linear(input_size, hidden_size)
        self.relu1 = ReLU()

        self.fc2 = Linear(hidden_size, hidden_size)
        self.relu2 = ReLU()

        self.fc3 = Linear(hidden_size, output_size)
        # 注意：我们使用CrossEntropyLoss，所以最后一层不需要softmax

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def parameters(self):
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params


class IrisClassifier(Module):
    """鸢尾花分类器"""

    def __init__(self):
        super().__init__()
        # 更简单的网络
        self.fc1 = Linear(4, 10)
        self.relu = ReLU()
        self.fc2 = Linear(10, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测单个样本"""
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)

        # 转换为Tensor
        x_tensor = Tensor(x.reshape(1, -1), requires_grad=False)

        # 前向传播
        with self._no_grad():
            output = self.forward(x_tensor)

        # 获取预测结果
        pred = np.argmax(output.data, axis=1)
        return pred[0], output.data[0]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """批量预测"""
        X_tensor = Tensor(X.astype(np.float32), requires_grad=False)

        with self._no_grad():
            outputs = self.forward(X_tensor)

        predictions = np.argmax(outputs.data, axis=1)
        probabilities = self._softmax(outputs.data)

        return predictions, probabilities

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """手动实现softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _no_grad(self):
        """上下文管理器，临时禁用梯度计算"""

        class NoGradContext:
            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        return NoGradContext()

    def evaluate(self, test_loader):
        """评估模型"""
        total_correct = 0
        total_samples = 0

        for batch_x, batch_y in test_loader:
            outputs = self.forward(batch_x)
            predictions = np.argmax(outputs.data, axis=1)
            labels = batch_y.data

            total_correct += np.sum(predictions == labels)
            total_samples += len(labels)

        accuracy = total_correct / total_samples
        return accuracy


def create_iris_model(model_type='simple'):
    """创建鸢尾花分类模型"""
    if model_type == 'simple':
        return IrisClassifier()
    elif model_type == 'deep':
        return IrisSimpleNN()
    else:
        raise ValueError(f"未知的模型类型: {model_type}")