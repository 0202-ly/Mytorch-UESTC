# mytorch/iris_dataset.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from .dataset import Dataset
from .tensor import Tensor


class IrisDataset(Dataset):
    """鸢尾花数据集"""

    def __init__(self, data_root=None, train=True, test_size=0.2, random_state=42):
        super().__init__()

        # 加载鸢尾花数据
        iris = load_iris()
        X = iris.data
        y = iris.target

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if train:
            self.data = X_train
            self.labels = y_train
        else:
            self.data = X_test
            self.labels = y_test

        # 标准化特征
        self._normalize()

    def _normalize(self):
        """标准化数据"""
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        self.data = (self.data - mean) / (std + 1e-8)

    def __getitem__(self, item):
        # 返回 Tensor 对象
        features = Tensor(self.data[item].astype(np.float32), requires_grad=False)
        label = Tensor(np.array(self.labels[item]).astype(np.int64), requires_grad=False)
        return features, label

    def __len__(self):
        return len(self.data)


class IrisDataFrameDataset(Dataset):
    """从CSV文件加载鸢尾花数据"""

    def __init__(self, csv_path=None, train=True, test_size=0.2):
        super().__init__()

        if csv_path:
            # 从CSV文件加载
            df = pd.read_csv(csv_path)
        else:
            # 从sklearn加载并转换为DataFrame
            iris = load_iris()
            df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            df['target'] = iris.target

        # 划分数据集
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        if train:
            self.df = train_df
        else:
            self.df = test_df

        # 分离特征和标签
        self.features = self.df.iloc[:, :-1].values.astype(np.float32)
        self.labels = self.df.iloc[:, -1].values.astype(np.int64)

        # 标准化
        self._normalize()

    def _normalize(self):
        mean = np.mean(self.features, axis=0)
        std = np.std(self.features, axis=0)
        self.features = (self.features - mean) / (std + 1e-8)

    def __getitem__(self, item):
        features = Tensor(self.features[item], requires_grad=False)
        label = Tensor(np.array(self.labels[item]), requires_grad=False)
        return features, label

    def __len__(self):
        return len(self.features)