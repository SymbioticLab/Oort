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

    def __init__(self, data, sizes=None, sequential=0, rationOfClassWorker=None, filter_class=0, seed=10, args = {'balanced_client':0, 'param': 1.5}):
        if sizes is None:
            sizes = [0.7, 0.2, 0.1]

        self.data = data
        self.partitions = []
        targets = {}

        rng = Random()
        rng.seed(seed)
        np.random.seed(seed)
        totalSamples = 0
        usedSamples = 50000

        # categarize the samples
        for index, (inputs, target) in enumerate(data):
            if target not in targets:
                targets[target] = []
            targets[target].append(index)
            totalSamples += 1

        keyDir = {key:i for i, key in enumerate(targets.keys())}
        keyLength = [len(targets[key]) for key in targets.keys()]

        self.classPerWorker = np.zeros([len(sizes), len(list(targets.keys()))])
        data_len = len(data)

        fstat = open("/tmp/log", "a")
        if sequential == 0:
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)

            for ratio in sizes:
                part_len = int(ratio * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

        else:
            fstat.writelines('========= Start of Class/Worker =========\n')

            # deal with the balanced dataset
            if args['balanced_client'] > 0:
                balanced_class_len = int(sizes[i] * data_len/len(targets.keys()))

                for i in range(args['balanced_client']):
                    balacned_class_set = []

                    for key in targets:
                        balacned_class_set += targets[key][:balanced_class_len]
                        targets[key] = targets[key][balanced_class_len:]

                    self.partitions.append(balacned_class_set)
                    rng.shuffle(self.partitions[-1])

                    fstat.writelines(repr([balanced_class_len for i in range(len(targets.keys()))]) + '\n')

                sizes = sizes[args['balanced_client']:]

            if rationOfClassWorker is None:
                # random distribution
                if sequential == 1:
                    rationOfClassWorker = np.random.rand(len(sizes), len(targets.keys()))
                # zipf distribution
                elif sequential == 2:
                    rationOfClassWorker = np.random.zipf(args['param'], [len(sizes), len(targets.keys())])
                    fstat.writelines("==== Load Zipf Distribution ====\n {} \n".format(repr(rationOfClassWorker)))
                    rationOfClassWorker = rationOfClassWorker.astype(np.float32)
                else:
                    rationOfClassWorker = np.ones((len(sizes), len(targets.keys()))).astype(np.float32)

            if filter_class > 0:
                for w in range(len(sizes)):
                    # randomly filter classes
                    wrandom = rng.sample(range(len(targets.keys())), filter_class)
                    for wr in wrandom:
                        rationOfClassWorker[w][wr] = 0.001

            # normalize the ratios
            if sequential == 1 or sequential == 3:
                sumRatiosPerClass = np.sum(rationOfClassWorker, axis=1)
                for worker in range(len(sizes)):
                    rationOfClassWorker[worker, :] = rationOfClassWorker[worker, :]/float(sumRatiosPerClass[worker])


                # split the classes
                for worker in range(len(sizes)):
                    self.partitions.append([])
                    # enumerate the ratio of classes it should take
                    for c in list(targets.keys()):
                        takeLength = min(floor(usedSamples * rationOfClassWorker[worker][keyDir[c]]), keyLength[keyDir[c]])
                        rng.shuffle(targets[c])
                        self.partitions[-1] += targets[c][0:takeLength]
                        self.classPerWorker[worker][keyDir[c]] += takeLength
                        #targets[c] = targets[c][takeLength:]

                    rng.shuffle(self.partitions[-1])
            else:
                sumRatiosPerClass = np.sum(rationOfClassWorker, axis=0)
                for c in targets.keys():
                    rationOfClassWorker[:, keyDir[c]] = rationOfClassWorker[:, keyDir[c]]/float(sumRatiosPerClass[keyDir[c]])

                # split the classes
                for worker in range(len(sizes)):
                    self.partitions.append([])
                    # enumerate the ratio of classes it should take
                    for c in list(targets.keys()):
                        takeLength = floor(keyLength[keyDir[c]] * rationOfClassWorker[worker][keyDir[c]])
                        self.partitions[-1] += targets[c][0:takeLength]
                        self.classPerWorker[worker][keyDir[c]] += takeLength
                        targets[c] = targets[c][takeLength:]

                    rng.shuffle(self.partitions[-1])


        # log
        fstat.writelines(repr(self.classPerWorker) + '\n')
        fstat.writelines('========= End of Class/Worker =========\n')
        fstat.close()

        print("====Total samples {}, Label types {}, with {} \n".format(totalSamples, len(targets.keys()), repr(keyLength)))

    def use(self, partition, istest):

        fmetrics = open('/tmp/sampleDistribution', 'a')
        totalLabels = [0 for i in range(len(self.classPerWorker[0]))]

        for index, row in enumerate(self.classPerWorker):
            rowStr = ''
            numSamples = 0
            for i, label in enumerate(self.classPerWorker[index]):
                rowStr += '\t'+str(int(label))
                totalLabels[i] += label
                numSamples += label

            fmetrics.writelines(str(index) + ':\t' + rowStr + '\n' + 'with sum:\t' + str(numSamples) + '\t' + repr(len(self.partitions[index]))+'\n')
            fmetrics.writelines("=====================================\n")

        fmetrics.writelines("Total selected samples is: {}, with {}\n".format(str(sum(totalLabels)), repr(totalLabels)))
        fmetrics.writelines("=====================================\n")

        fmetrics.close()

        _partition = -1 if istest else partition

        print("====Data length is {}".format(len(self.partitions[_partition])))
        return Partition(self.data, self.partitions[_partition])

def partition_dataset(dataset, workers, partitionRatio=[], sequential=0, rationOfClassWorker=None, filter_class=0):
    """ Partitioning Data """
    workers_num = len(workers)
    partition_sizes = [1.0 / workers_num for _ in range(workers_num)]

    if len(partitionRatio) > 0:
        partition_sizes = partitionRatio

    partition = DataPartitioner(data=dataset, sizes=partition_sizes, sequential=sequential, rationOfClassWorker=rationOfClassWorker,filter_class=filter_class)
    return partition


def select_dataset(workers: list, rank: int, partition: DataPartitioner, batch_size: int, istest=False):
    workers_num = len(workers)
    partition_dict = {workers[i]: i for i in range(workers_num)}
    partition = partition.use(partition_dict[rank], istest)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
