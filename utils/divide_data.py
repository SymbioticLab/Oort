# -*- coding: utf-8 -*-

from random import Random

from torch.utils.data import DataLoader


class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=None, seed=1234):
        if sizes is None:
            sizes = [0.7, 0.2, 0.1]
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for ratio in sizes:
            part_len = int(ratio * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(dataset, workers):
    """ Partitioning Data """
    workers_num = len(workers)
    partition_sizes = [1.0 / workers_num for _ in range(workers_num)]
    partition = DataPartitioner(dataset, partition_sizes)
    return partition


def select_dataset(workers: list, rank: int, partition: DataPartitioner, batch_size: int):
    workers_num = len(workers)
    partition_dict = {workers[i]: i for i in range(workers_num)}
    partition = partition.use(partition_dict[rank])
    return DataLoader(partition, batch_size=batch_size, shuffle=True)
