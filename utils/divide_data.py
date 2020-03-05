# -*- coding: utf-8 -*-
from random import Random
from core.dataloader import DataLoader
import numpy as np
from math import *
import logging
from scipy import stats
import numpy as np
from pyemd import emd
from collections import OrderedDict

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
    def __init__(self, data, numOfClass=0, seed=10, splitConfFile=None, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)
        self.data = data
        np.random.seed(seed)

        self.targets = OrderedDict()
        self.indexToLabel = {}
        self.totalSamples = 0
        self.data_len = len(self.data)

        if isTest:
            # randomly generate some
            # categarize the samples
            self.targets[0] = []
            for index in range(self.data_len):
                self.targets[0].append(index)
                self.indexToLabel[index] = 0

            self.totalSamples += self.data_len

        elif splitConfFile is None:
            # categarize the samples
            for index, (inputs, label) in enumerate(self.data):
                if label not in self.targets:
                    self.targets[label] = []
                self.targets[label].append(index)
                self.indexToLabel[index] = label
            
            self.totalSamples += len(self.data)
        else:
            # each row denotes the number of samples in this class
            with open(splitConfFile, 'r') as fin:
                labelSamples = [int(x.strip()) for x in fin.readlines()]

            # categarize the samples
            baseIndex = 0
            for label, _samples in enumerate(labelSamples):
                for k in range(_samples):
                    self.indexToLabel[baseIndex + k] = label
                self.targets[label] = [baseIndex + k for k in range(_samples)]
                self.totalSamples += _samples
                baseIndex += _samples

        self.numOfLabels = max(len(self.targets.keys()), numOfClass)
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
        keyDir = {key:int(key) for i, key in enumerate(targets.keys())}
        keyLength = [0] * numOfLabels

        for key in keyDir.keys():
            keyLength[keyDir[key]] = len(targets[key])

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])
        
        # random partition
        if sequential == 0:
            logging.info("========= Start of Random Partition =========\n")
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
                    # randomly filter classes by forcing zero samples
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
            elif sequential == 2:
                sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=0)
                for c in targets.keys():
                    ratioOfClassWorker[:, keyDir[c]] = ratioOfClassWorker[:, keyDir[c]]/float(sumRatiosPerClass[keyDir[c]])

                # split the classes
                for worker in range(len(sizes)):
                    self.partitions.append([])
                    # enumerate the ratio of classes it should take
                    for c in list(targets.keys()):
                        takeLength = min(int(math.ceil(keyLength[keyDir[c]] * ratioOfClassWorker[worker][keyDir[c]])), len(targets[c]))
                        self.partitions[-1] += targets[c][0:takeLength]
                        tempClassPerWorker[worker][keyDir[c]] += takeLength
                        targets[c] = targets[c][takeLength:]

                    self.rng.shuffle(self.partitions[-1])

            elif sequential == 4:
                # load data from given config file
                clientGivenSamples = {}
                with open(args['clientSampleConf'], 'r') as fin:
                    for clientId, line in enumerate(fin.readlines()):
                        clientGivenSamples[clientId] = [int(x) for x in line.strip().split()]

                # split the data
                for clientId in range(len(clientGivenSamples.keys())):
                    self.partitions.append([])

                    for c in list(targets.keys()):
                        takeLength = clientGivenSamples[clientId][c]
                        if clientGivenSamples[clientId][c] > targets[c]:
                            logging.info("========== Failed to allocate {} samples for class {} to client {}, actual quota is {}"\
                                .format(clientGivenSamples[clientId][c], c, clientId, targets[c]))
                            takeLength = targets[c]

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
        _partition = partition
        resultIndex = self.partitions[_partition]
        self.rng.shuffle(resultIndex)

        logging.info("====Data length for client {} is {}".format(partition, len(resultIndex)))
        return Partition(self.data, resultIndex)

    def getDistance(self):
        return self.workerDistance

    def getSize(self):
        # return the size of samples
        return [len(partition) for partition in self.partitions]

def partition_dataset(partitioner, workers, partitionRatio=[], sequential=0, ratioOfClassWorker=None, filter_class=0, arg={'param': 1.95}):
    """ Partitioning Data """
    workers_num = len(workers)
    partition_sizes = [1.0 / workers_num for _ in range(workers_num)]

    if len(partitionRatio) > 0:
        partition_sizes = partitionRatio

    partitioner.partitionData(sizes=partition_sizes, sequential=sequential, ratioOfClassWorker=ratioOfClassWorker,filter_class=filter_class, args=arg)

def select_dataset(rank: int, partition: DataPartitioner, batch_size: int, isTest=False):
    partition = partition.use(rank - 1, isTest)

    #if istest:
    #    return DataLoader(partition, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=16, drop_last=True)
    #else:
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=10, drop_last=True)
