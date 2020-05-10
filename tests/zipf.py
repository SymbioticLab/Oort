import random
import numpy as np
import math
import argparse
import dit,numpy as np
from dit.divergences import jensen_shannon_divergence

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=20)
parser.add_argument('--totalWorkers', type=int, default=100)
parser.add_argument('--sampledWorkers', type=int, default=20)
parser.add_argument('--totalClass', type=int, default=47)
parser.add_argument('--zipfAlpha', type=float, default=1.5)
parser.add_argument('--samplesPerClass', type=int, default=50000)
parser.add_argument('--sample_seed', type=int, default=0)
parser.add_argument('--same_size', type=int, default=0)
parser.add_argument('--client_size', type=int, default=25000)

args = parser.parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)

totalWorkers = args.totalWorkers
sampledWorkers = args.sampledWorkers
totalClass = args.totalClass
zipfAlpha = args.zipfAlpha
samplesPerClass = args.samplesPerClass

partitions = []
sizes = [i for i in range(totalWorkers)]
targets = {}

for c in range(totalClass):
    targets[c] = [i for i in range(samplesPerClass)]

keyDir = {key:i for i, key in enumerate(targets.keys())}
keyLength = [samplesPerClass for i in range(totalClass)]
filter_class = 0

rationOfClassWorker = np.random.zipf(zipfAlpha, [totalWorkers, 1]).astype(np.float32)

for i in range(totalClass-1):
    rationOfClassWorker = np.concatenate((rationOfClassWorker, np.random.zipf(zipfAlpha, [totalWorkers, 1]).astype(np.float32)), axis=1)

if filter_class > 0:
    for w in range(len(sizes)):
        # randomly filter classes
        wrandom = rng.sample(range(len(targets.keys())), filter_class)
        for wr in wrandom:
            rationOfClassWorker[w][wr] = 0.001

# normalize the ratios
if args.same_size == 0:
    sumRatiosPerClass = np.sum(rationOfClassWorker, axis=0)
    for c in targets.keys():
        rationOfClassWorker[:, keyDir[c]] = rationOfClassWorker[:, keyDir[c]]/float(sumRatiosPerClass[keyDir[c]])
else:
    sumRatiosPerClass = np.sum(rationOfClassWorker, axis=1)
    for w in range(len(sizes)):
        rationOfClassWorker[w, :] = rationOfClassWorker[w, :]/float(sumRatiosPerClass[w])

classPerWorker = np.zeros((totalWorkers, totalClass)).astype(np.int)
#print (rationOfClassWorker)
# split the classes
for worker in range(len(sizes)):
    #partitions.append([])
    # enumerate the ratio of classes it should take
    for c in targets.keys():
        if args.same_size == 0:
            takeLength = int(math.floor(keyLength[keyDir[c]] * rationOfClassWorker[worker][keyDir[c]]))
        else:
            takeLength = int(math.floor(args.client_size * rationOfClassWorker[worker][keyDir[c]]))
        #partitions[-1] += targets[c][0:takeLength]
        classPerWorker[worker][keyDir[c]] += takeLength
        #targets[c] = targets[c][takeLength:]

    #rng.shuffle(partitions[-1])

# for worker in range(len(classPerWorker)):
#     print("For worker {}, total samples is {}, with {} \n".format(worker, sum(classPerWorker[worker]), repr(classPerWorker[worker])))
#     print("===================================")


# for sampling a subset
def randomSampling(seeds = args.sample_seed):
    random.seed(seeds)
    np.random.seed(seeds)

    sampledIndice = random.sample(range(totalWorkers), sampledWorkers)
    samples = [0 for i in range(totalClass)]

    for i, index in enumerate(sampledIndice):
        for k, c in enumerate(classPerWorker[index]):
            samples[k] += c
        #print("For worker {}, global index {},  total samples is {}, with {} \n".format(i, index, sum(classPerWorker[index]), repr(classPerWorker[index])))
        #print("===================================")

    print("===================================")
    print("Total samples sum {}, \n {}".format(sum(samples), repr(samples)))

def measureDistance(overall, individual):
    # Caculates Jensen-Shannon Divergence for each worker
    tempDataSize = sum(individual)
    tempDistr = dit.ScalarDistribution(np.arange(len(individual)), [c / float(tempDataSize) for c in individual])
    #tempDistr = dit.ScalarDistribution(np.arange(len(individual)), [c / float(max(individual)) for c in individual])
    return jensen_shannon_divergence([overall, tempDistr])

def isSampleEnough(target, current):
    for i, item in enumerate(target):
        if item > current[i]:
            return False

    return True

#randomSampling(i + args.sample_seed*10)

# get the overall -> target
target = [samplesPerClass * 0.4 for i in range(totalClass)]
totalDataSize = sum(target)
representative = dit.ScalarDistribution(np.arange(len(target)), [key / float(totalDataSize) for key in target])
#representative = dit.ScalarDistribution(np.arange(len(target)), [key / float(max(target)) for key in target])

distanceDic = []
for worker in range(totalWorkers):
    distanceDic.append(measureDistance(representative, classPerWorker[worker]))

# Sort the distance
index = sorted(range(len(distanceDic)), key=lambda k: distanceDic[k])
# print (distanceDic)
# print (index)

# take the clients until reach the target
sampledSamples = [0 for i in range(totalClass)]
sampledClients = []
sampledDistance = []

for w in range(totalWorkers):
    print (classPerWorker[w], distanceDic[w])

for work in index:
    if not isSampleEnough(target, sampledSamples):
        for i, item in enumerate(classPerWorker[work]):
            sampledSamples[i] += item
        sampledDistance.append(distanceDic[work])
        sampledClients.append(work)
        print('After picking {}, now the bin is {}'.format(classPerWorker[work], sampledSamples))

print("With sampling target: \n{}\nSolver picks {} clients, avg distance is {} \nFinal # of sampled data is:\n{}\n".format(target, len(sampledClients), sum(sampledDistance)/float(len(sampledDistance)), sampledSamples))
    

