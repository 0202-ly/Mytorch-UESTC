# mytorch/async_dataloader.py
"""
异步数据加载器
支持多线程预取、生产者-消费者模式
"""
import threading
import queue
import time
from typing import Iterator, Optional, List, Any
import numpy as np

from mytorch.dataloader import Dataloader
from mytorch.dataset import Dataset


class AsyncDataLoader:
    """
    异步数据加载器
    使用生产者-消费者模式，在后台线程中预取数据
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 num_workers: int = 2,
                 prefetch_factor: int = 4,
                 collate_fn=None,
                 timeout: float = 30.0):
        """
        初始化异步数据加载器

        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作线程数
            prefetch_factor: 预取因子（队列大小）
            collate_fn: 批次整理函数
            timeout: 数据获取超时时间（秒）
        """
        # 创建基础DataLoader作为数据源
        self.base_loader = Dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout

        # 创建队列
        self.queue = queue.Queue(maxsize=prefetch_factor)

        # 控制标志
        self._stop_event = threading.Event()
        self._workers = []
        self._is_running = False

        # 统计信息
        self.stats = {
            'total_batches': 0,
            'queue_size': 0,
            'wait_time': 0,
            'worker_stats': []
        }

    def _worker(self, worker_id: int):
        """工作线程函数"""
        worker_stats = {
            'worker_id': worker_id,
            'batches_produced': 0,
            'total_time': 0,
            'wait_time': 0
        }

        try:
            for batch in self.base_loader:
                if self._stop_event.is_set():
                    break

                start_time = time.time()
                self.queue.put(batch, timeout=1.0)
                worker_stats['batches_produced'] += 1
                worker_stats['total_time'] += time.time() - start_time

            # 发送结束信号
            self.queue.put(None)

        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            self.queue.put(None)
        finally:
            self.stats['worker_stats'].append(worker_stats)

    def _start_workers(self):
        """启动工作线程"""
        if self._is_running:
            return

        self._is_running = True
        self._stop_event.clear()

        # 启动工作线程
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker, args=(i,))
            t.daemon = True
            t.start()
            self._workers.append(t)

    def _stop_workers(self):
        """停止工作线程"""
        self._stop_event.set()

        # 等待所有线程结束
        for t in self._workers:
            t.join(timeout=2.0)

        self._workers.clear()
        self._is_running = False

    def __iter__(self):
        """返回迭代器"""
        self._start_workers()
        return self

    def __next__(self):
        """获取下一个批次"""
        if not self._is_running:
            self._start_workers()

        try:
            batch = self.queue.get(timeout=self.timeout)

            if batch is None:
                # 收到结束信号
                self._stop_workers()
                raise StopIteration

            self.stats['total_batches'] += 1
            self.stats['queue_size'] = self.queue.qsize()

            return batch

        except queue.Empty:
            self._stop_workers()
            raise StopIteration

    def __len__(self):
        return len(self.base_loader)

    def reset(self):
        """重置加载器"""
        self._stop_workers()
        self.base_loader.reset()

        # 清空队列
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def get_stats(self):
        """获取统计信息"""
        self.stats['queue_size'] = self.queue.qsize()
        return self.stats


class ParallelDataLoader:
    """
    并行数据加载器
    使用多进程进行数据加载（适用于CPU密集型预处理）
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 collate_fn=None):
        """
        初始化并行数据加载器

        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            prefetch_factor: 预取因子
            collate_fn: 批次整理函数
        """
        # 注意：完整的多进程实现需要使用 multiprocessing
        # 这里提供框架，实际实现较复杂
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.collate_fn = collate_fn

        # 使用异步版本作为基础
        self.async_loader = AsyncDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(num_workers, 4),  # 限制线程数
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn
        )

    def __iter__(self):
        return iter(self.async_loader)

    def __next__(self):
        return next(self.async_loader)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def reset(self):
        self.async_loader.reset()

    def get_stats(self):
        return self.async_loader.get_stats()