# mytorch/dataset/iris_dataset.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mytorch import Dataset
from mytorch import Tensor

class IrisDataset(Dataset):
    """
    鸢尾花数据集
    修复：标准化时统一使用训练集的统计量，防止测试集数据泄露。
    """

    def __init__(self, train=True, test_size=0.2, random_state=42):
        super().__init__()

        # 1. 加载数据
        iris = load_iris()
        X = iris.data
        y = iris.target

        # 2. 划分训练集和测试集
        # 注意：这里必须固定 random_state，确保 train=True 和 train=False 时划分的数据是一致的
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 3. 【关键修复】计算归一化统计量 (仅基于训练集)
        # 即使当前是测试集模式，我们也必须用训练集的均值和方差来标准化，
        # 模拟真实场景中我们不知道测试集分布的情况。
        self.train_mean = np.mean(X_train, axis=0)
        self.train_std = np.std(X_train, axis=0)

        # 4. 根据模式选择数据
        if train:
            self.raw_data = X_train
            self.labels = y_train
        else:
            self.raw_data = X_test
            self.labels = y_test

        # 5. 执行标准化
        self.data = self._normalize(self.raw_data)

    def _normalize(self, data):
        """使用训练集的统计量进行标准化"""
        return (data - self.train_mean) / (self.train_std + 1e-8)

    def __getitem__(self, item):
        # 返回 Tensor 对象
        # features: (4,) -> Tensor
        features = Tensor(self.data[item].astype(np.float32), requires_grad=False)
        # label: scalar -> Tensor
        label = Tensor(np.array(self.labels[item]).astype(np.int64), requires_grad=False)
        return features, label

    def __len__(self):
        return len(self.data)


class IrisDataFrameDataset(Dataset):
    """
    从CSV文件加载鸢尾花数据 (支持 Pandas)
    同样修复了标准化逻辑
    """

    def __init__(self, csv_path=None, train=True, test_size=0.2, random_state=42):
        super().__init__()

        # 1. 加载数据到 DataFrame
        if csv_path:
            df = pd.read_csv(csv_path)
        else:
            # 从sklearn加载并模拟为DataFrame
            iris = load_iris()
            df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            df['target'] = iris.target

        # 2. 提取特征和标签
        # 假设最后一列是标签，前面是特征
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)

        # 3. 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 4. 【关键修复】计算训练集统计量
        self.train_mean = np.mean(X_train, axis=0)
        self.train_std = np.std(X_train, axis=0)

        # 5. 选择数据
        if train:
            self.raw_features = X_train
            self.labels = y_train
        else:
            self.raw_features = X_test
            self.labels = y_test

        # 6. 标准化
        self.features = self._normalize(self.raw_features)

    def _normalize(self, data):
        """使用训练集统计量"""
        return (data - self.train_mean) / (self.train_std + 1e-8)

    def __getitem__(self, item):
        features = Tensor(self.features[item], requires_grad=False)
        label = Tensor(np.array(self.labels[item]), requires_grad=False)
        return features, label

    def __len__(self):
        return len(self.features)