import math

import numpy as np

from .dataset import Dataset
from .tensor import Tensor


class Dataloader:
    def __init__(self,
                 dataset:Dataset,
                 collate_fn,
                 batch_size = 1,
                 shuffle = True,
                 ):
        self.dataset = dataset
        if collate_fn is None:
            collate_fn = self.default_collate
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_its = math.ceil(self.data_size / batch_size)
        self.it = 0
        self.indices = None
        self.reset()

    def reset(self):
        self.it = 0  # 重置迭代计数器
        if self.shuffle:
            # 生成随机排列的索引
            self.indices = np.random.permutation(self.data_size)
        else:
            # 生成顺序索引
            self.indices = np.arange(self.data_size)

    def __len__(self):
        return self.max_its

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.it >= self.max_its:
            self.reset()
            raise StopIteration
        start_idx = self.it * self.batch_size
        end_idx = min((self.it + 1) * self.batch_size, self.data_size)
        batch_indices = self.indices[start_idx:end_idx]

        # 收集批次数据
        batch = [self.dataset[i] for i in batch_indices]
        self.it += 1

        return self.collate_fn(batch)


    def default_collate(self, batch):
        if isinstance(batch[0], tuple):
            # 转置批次，从[(x1,y1), (x2,y2)...]变成([x1,x2...], [y1,y2...])
            transposed = zip(*batch)
            return [
                Tensor(np.stack([sample.data if isinstance(sample, Tensor) else sample
                                 for sample in samples]))
                for samples in transposed
            ]

        if isinstance(batch[0], Tensor):
            return Tensor(np.stack([x.data for x in batch]))

        return Tensor(np.array(batch))

