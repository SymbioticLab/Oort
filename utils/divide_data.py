# -*- coding: utf-8 -*-

from random import Random
from torch.utils.data import DataLoader
import numpy as np
from math import *

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

    def __init__(self, data, sizes=None, sequential=0, rationOfClassWorker=None, filter_class=0, seed=1234):
        if sizes is None:
            sizes = [0.7, 0.2, 0.1]

        self.data = data
        self.partitions = []
        targets = {}

        rng = Random()
        rng.seed(seed)
        np.random.seed(seed)

        # categarize the samples
        for index, (inputs, target) in enumerate(data):
            if target not in targets:
                targets[target] = []
            targets[target].append(index)

        classPerWorker = np.zeros([len(sizes), len(list(targets.keys()))])

        if sequential == 0:
            data_len = len(data)
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)
            
            for ratio in sizes:
                part_len = int(ratio * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

        elif sequential == 1:

            keyDir = {key:i for i, key in enumerate(list(targets.keys()))}
            keyLength = [len(targets[key]) for key in list(targets.keys())]

            if rationOfClassWorker is None:
                rationOfClassWorker = np.random.rand(len(sizes), len(list(targets.keys())))

            if filter_class > 0:
                for w in range(len(sizes)):
                    # randomly filter classes
                    wrandom = rng.sample(range(len(list(targets.keys()))), filter_class)
                    for wr in wrandom:
                        rationOfClassWorker[w][wr] = 0.001

            # normalize the ratios
            sumRatiosPerClass = np.sum(rationOfClassWorker, axis=0)
            for c in list(targets.keys()):
                rationOfClassWorker[:, c] = rationOfClassWorker[:, c]/float(sumRatiosPerClass[c])

            # split the classes
            for worker in range(len(sizes)):
                self.partitions.append([])
                # enumerate the ratio of classes it should take
                for c in list(targets.keys()):
                    takeLength = floor(keyLength[keyDir[c]] * rationOfClassWorker[worker][keyDir[c]])
                    self.partitions[worker] += targets[c][0:takeLength]
                    classPerWorker[worker][keyDir[c]] += takeLength
                    targets[c] = targets[c][takeLength:]

                rng.shuffle(self.partitions[-1])

            # log 
            fstat = open("/tmp/log", "a")
            fstat.writelines(repr(classPerWorker) + '\n')
            fstat.close()

    def use(self, partition):
        print("====Data length is {}".format(len(self.partitions[partition])))
        return Partition(self.data, self.partitions[partition])

def partition_dataset(dataset, workers, partitionRatio=[], sequential=0, rationOfClassWorker=None, filter_class=0):
    """ Partitioning Data """
    workers_num = len(workers)
    partition_sizes = [1.0 / workers_num for _ in range(workers_num)]

    if len(partitionRatio) > 0:
        partition_sizes = partitionRatio
        
    partition = DataPartitioner(data=dataset, sizes=partition_sizes, sequential=sequential, rationOfClassWorker=rationOfClassWorker,filter_class=filter_class)
    return partition


def select_dataset(workers: list, rank: int, partition: DataPartitioner, batch_size: int):
    workers_num = len(workers)
    partition_dict = {workers[i]: i for i in range(workers_num)}
    partition = partition.use(partition_dict[rank])
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
