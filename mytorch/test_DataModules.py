import numpy as np

from .dataset import Dataset
from .dataloader import Dataloader
from .tensor import Tensor


class TupleDataset(Dataset):
    """返回 (特征, 标签) 元组的数据集，用于测试元组类型的 collate"""
    def __init__(self, num_samples: int, feature_dim: int = 3):
        # 特征：随机生成 (num_samples, feature_dim) 的数据（部分转 Tensor）
        self.features = [
            Tensor(np.random.randn(feature_dim)) if i % 3 == 0 else np.random.randn(feature_dim)
            for i in range(num_samples)
        ]
        # 标签：0/1 二分类（部分转 Tensor）
        self.labels = [
            Tensor(0) if i % 2 == 0 else 1
            for i in range(num_samples)
        ]

    def __getitem__(self, item):
        return self.features[item], self.labels[item]  # 返回 (特征, 标签) 元组

    def __len__(self):
        return len(self.features)

def test_tuple_with_shuffle():
    print("="*50)
    print("测试2：(特征,标签) 元组数据集 + batch_size=2 + shuffle=True")
    # 初始化数据集（5个样本，特征维度3）
    dataset = TupleDataset(num_samples=5, feature_dim=3)
    # 初始化 Dataloader（打乱，batch=2）
    dataloader = Dataloader(dataset, None, batch_size=2, shuffle=True)

    # 验证总批次数：5/2=2.5 → 向上取整为3
    print(f"预期总批次数：3，实际：{len(dataloader)}")

    # 遍历取 batch，验证每个 batch 的结构
    for idx, batch in enumerate(dataloader):
        print(f"\n第{idx}批：")
        print(f"batch 类型：{type(batch)}")  # 预期是列表（[特征Tensor, 标签Tensor]）
        print(f"特征形状：{batch[0].data.shape}")  # 预期：前2批 (2,3)，最后1批 (1,3)
        print(f"标签形状：{batch[1].data.shape}")  # 预期：前2批 (2,)，最后1批 (1,)
        print(f"特征数据：{batch[0]}")
        print(f"标签数据：{batch[1]}")

    print("测试2结束\n")

# 执行测试
test_tuple_with_shuffle()
