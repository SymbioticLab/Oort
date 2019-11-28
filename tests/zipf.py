import random
import numpy as np

random.seed(12)
np.random.seed(12)

sizes = [i for i in range(10)]
#keyDir = {key:i for i, key in enumerate(list(targets.keys()))}
keyLength = [131600/47 for i in range(47)]
filter_class = 0
rationOfClassWorker = np.random.zipf(2, [9, 47]).astype(np.float32)

if filter_class > 0:
    for w in range(len(sizes)):
        # randomly filter classes
        wrandom = rng.sample(range(len(list(targets.keys()))), filter_class)
        for wr in wrandom:
            rationOfClassWorker[w][wr] = 0.001

# normalize the ratios
sumRatiosPerClass = np.sum(rationOfClassWorker, axis=0)
for c in range(47):
    rationOfClassWorker[:, c] = rationOfClassWorker[:, c]/float(sumRatiosPerClass[c])# * 131600/47

print (rationOfClassWorker)
# split the classes
# for worker in range(len(sizes)):
#     partitions.append([])
#     # enumerate the ratio of classes it should take
#     for c in list(targets.keys()):
#         takeLength = floor(keyLength[keyDir[c]] * rationOfClassWorker[worker][keyDir[c]])
#         partitions[worker] += targets[c][0:takeLength]
#         classPerWorker[worker][keyDir[c]] += takeLength
#         targets[c] = targets[c][takeLength:]

    #rng.shuffle(self.partitions[-1])
