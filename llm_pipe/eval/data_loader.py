import random
from typing import List, Dict
from collections import deque

class DataLoader:
    def __init__(self, data: List[Dict], batch_size: int=32, shuffle: bool=False):
        """
        初始化数据加载器
        
        :param data: 字典列表形式的数据
        :param batch_size: 每个批次的数据数量
        :param shuffle: 是否在每个epoch开始时打乱数据
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = list(range(len(data)))
        self._reset_indexes()

    def _reset_indexes(self):
        """
        重置索引，若需要打乱数据则进行打乱操作
        """
        if self.shuffle:
            random.shuffle(self.indexes)
        self.current_idx = 0

    def __iter__(self):
        """
        迭代器开始时重置索引
        """
        self._reset_indexes()
        return self

    def __next__(self):
        """
        获取下一个批次的数据
        """
        if self.current_idx >= len(self.data):
            raise StopIteration

        batch_indexes = self.indexes[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        batch = [self.data[i] for i in batch_indexes]
        return batch[0] if len(batch) == 1 else batch
