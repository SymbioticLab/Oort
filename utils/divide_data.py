# -*- coding: utf-8 -*-
from random import Random
from torch.utils.data import DataLoader
import numpy as np
from math import *
import logging
from scipy import stats
import numpy as np
from pyemd import emd

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

    # len(sizes) is the number of workers
    # sequential 1-> random 2->zipf 3-> identical
    def __init__(self, data, seed=10):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)
        self.data = data
        np.random.seed(seed)


        self.targets = {}
        self.indexToLabel = {}
        self.totalSamples = 0

        # categarize the samples
        for index, (inputs, label) in enumerate(self.data):
            if label not in self.targets:
                self.targets[label] = []
            self.targets[label].append(index)
            self.indexToLabel[index] = label
            self.totalSamples += 1

        self.data_len = len(self.data)
        self.numOfLabels = len(self.targets.keys())
        self.workerDistance = []
        self.classPerWorker = None

    def getTargets(self):
        tempTarget = self.targets.copy()

        for key in tempTarget:
            self.rng.shuffle(tempTarget[key])

        return tempTarget
        
    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    # Calculates JSD between pairs of distribution
    def js_distance(self, x, y):
        m = (x + y)/2
        js = 0.5 * stats.entropy(x, m) + 0.5 * stats.entropy(y, m)
        return js

    # Caculates Jensen-Shannon Divergence for each worker
    def get_JSD(self, dataDistr, tempClassPerWorker, sizes):
        for worker in range(len(sizes)):
            tempDataSize = sum(tempClassPerWorker[worker])
            if tempDataSize == 0:
                continue
            tempDistr =np.array([c / float(tempDataSize) for c in tempClassPerWorker[worker]])
            self.workerDistance.append(self.js_distance(dataDistr, tempDistr))

    # Generates a distance matrix for EMD
    def generate_distance_matrix(self, size):
        return np.logical_xor(1, np.identity(size)) * 1.0

    # Caculates Earth Mover's Distance for each worker
    def get_EMD(self, dataDistr, tempClassPerWorker, sizes):
        dist_matrix = self.generate_distance_matrix_v2(len(dataDistr))
        for worker in range(len(sizes)):
            tempDataSize = sum(tempClassPerWorker[worker])
            if tempDataSize == 0:
                continue
            tempDistr =np.array([c / float(tempDataSize) for c in tempClassPerWorker[worker]])
            self.workerDistance.append(emd(dataDistr, tempDistr, dist_matrix))

    def partitionData(self, sizes=None, sequential=0, ratioOfClassWorker=None, filter_class=0, args = None):
        targets = self.getTargets()
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()

        usedSamples = 100000
        keyDir = {key:i for i, key in enumerate(targets.keys())}
        keyLength = [len(targets[key]) for key in targets.keys()]

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])
        
        if sequential == 0:
            indexes = [x for x in range(0, data_len)]
            self.rng.shuffle(indexes)

            for ratio in sizes:
                part_len = int(ratio * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

            for id, partition in enumerate(self.partitions):
                for index in partition:
                    tempClassPerWorker[id][self.indexToLabel[index]] += 1
        else:
            logging.info('========= Start of Class/Worker =========\n')

            # deal with the balanced dataset
            if args['balanced_client'] > 0:
                balanced_class_len = int(sizes[i] * data_len/numOfLabels)

                for i in range(args['balanced_client']):
                    balacned_class_set = []

                    for key in targets:
                        balacned_class_set += targets[key][:balanced_class_len]
                        targets[key] = targets[key][balanced_class_len:]

                    self.partitions.append(balacned_class_set)
                    self.rng.shuffle(self.partitions[-1])

                    logging.info(repr([balanced_class_len for i in range(numOfLabels)]) + '\n')

                sizes = sizes[args['balanced_client']:]

            if ratioOfClassWorker is None:
                # random distribution
                if sequential == 1:
                    ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels)
                # zipf distribution
                elif sequential == 2:
                    ratioOfClassWorker = np.random.zipf(args['param'], [len(sizes), numOfLabels])
                    logging.info("==== Load Zipf Distribution ====\n {} \n".format(repr(ratioOfClassWorker)))
                    ratioOfClassWorker = ratioOfClassWorker.astype(np.float32)
                else:
                    ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)

            if filter_class > 0:
                for w in range(len(sizes)):
                    # randomly filter classes
                    wrandom = self.rng.sample(range(numOfLabels), filter_class)
                    for wr in wrandom:
                        ratioOfClassWorker[w][wr] = 0.001

            # normalize the ratios
            if sequential == 1 or sequential == 3:
                sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)
                for worker in range(len(sizes)):
                    ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :]/float(sumRatiosPerClass[worker])
                # split the classes
                for worker in range(len(sizes)):
                    self.partitions.append([])
                    # enumerate the ratio of classes it should take
                    for c in list(targets.keys()):
                        takeLength = min(floor(usedSamples * ratioOfClassWorker[worker][keyDir[c]]), keyLength[keyDir[c]])
                        self.rng.shuffle(targets[c])
                        self.partitions[-1] += targets[c][0:takeLength]
                        tempClassPerWorker[worker][keyDir[c]] += takeLength

                    self.rng.shuffle(self.partitions[-1])
            else:
                sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=0)
                for c in targets.keys():
                    ratioOfClassWorker[:, keyDir[c]] = ratioOfClassWorker[:, keyDir[c]]/float(sumRatiosPerClass[keyDir[c]])

                # split the classes
                for worker in range(len(sizes)):
                    self.partitions.append([])
                    # enumerate the ratio of classes it should take
                    for c in list(targets.keys()):
                        takeLength = floor(keyLength[keyDir[c]] * ratioOfClassWorker[worker][keyDir[c]])
                        self.partitions[-1] += targets[c][0:takeLength]
                        tempClassPerWorker[worker][keyDir[c]] += takeLength
                        targets[c] = targets[c][takeLength:]

                    self.rng.shuffle(self.partitions[-1])

        # concatenate ClassPerWorker
        if self.classPerWorker is None:
            self.classPerWorker = tempClassPerWorker
        else:
            self.classPerWorker = np.concatenate((self.classPerWorker, tempClassPerWorker), axis=0)

        # Calculates statistical distances
        totalDataSize = sum(keyLength)
        # Overall data distribution
        dataDistr = np.array([key / float(totalDataSize) for key in keyLength])
        self.get_JSD(dataDistr, tempClassPerWorker, sizes)
            
        logging.info("Raw class per worker is : " + repr(tempClassPerWorker) + '\n')
        logging.info('========= End of Class/Worker =========\n')

    def log_selection(self):
        totalLabels = [0 for i in range(len(self.classPerWorker[0]))]
        logging.info("====Total # of workers is :{}, w/ {} labels, {}, {}".format(len(self.classPerWorker), len(self.classPerWorker[0]), len(self.partitions), len(self.workerDistance)))

        for index, row in enumerate(self.classPerWorker):
            rowStr = ''
            numSamples = 0
            for i, label in enumerate(self.classPerWorker[index]):
                rowStr += '\t'+str(int(label))
                totalLabels[i] += label
                numSamples += label

            logging.info(str(index) + ':\t' + rowStr + '\n' + 'with sum:\t' + str(numSamples) + '\t' + repr(len(self.partitions[index]))+ '\nDistance: ' + str(self.workerDistance[index])+ '\n')
            logging.info("=====================================\n")

        logging.info("Total selected samples is: {}, with {}\n".format(str(sum(totalLabels)), repr(totalLabels)))
        logging.info("=====================================\n")

    def use(self, partition, istest):
        _partition = -1 if istest else partition

        logging.info("====Data length for client {} is {}".format(partition, len(self.partitions[_partition])))
        self.rng.shuffle(self.partitions[_partition])
        return Partition(self.data, self.partitions[_partition])

    def getDistance(self):
        return self.workerDistance

    def getSize(self):
        # return the size of samples
        return [len(partition) for partition in self.partitions]

def partition_dataset(partitioner, workers, partitionRatio=[], sequential=0, ratioOfClassWorker=None, filter_class=0, arg={'balanced_client':0, 'param': 1.95}):
    """ Partitioning Data """
    workers_num = len(workers)
    partition_sizes = [1.0 / workers_num for _ in range(workers_num)]

    if len(partitionRatio) > 0:
        partition_sizes = partitionRatio

    partitioner.partitionData(sizes=partition_sizes, sequential=sequential, ratioOfClassWorker=ratioOfClassWorker,filter_class=filter_class, args=arg)

def select_dataset(rank: int, partition: DataPartitioner, batch_size: int, istest=False):
    #workers_num = len(workers)
    #partition_dict = {workers[i]: i for i in range(workers_num)}
    partition = partition.use(rank - 1, istest)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
