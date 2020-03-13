import math
from random import Random
from collections import OrderedDict

class UCB(object):

    def __init__(self, sample_seed):
        self.totalArms = OrderedDict()
        self.totalTries = 0
        self.alpha = 0.8

        self.exploration = 0.2
        self.exploitation = 1.0 - self.exploration

        self.rng = Random()
        self.rng.seed(sample_seed)

        self.orderedKeys = None

    def registerArm(self, armId, reward):
        # Initiate the score for arms. [score, # of tries]
        if armId not in self.totalArms:
            self.totalArms[armId] = [reward, 1, 0]

    def registerReward(self, armId, reward):
        self.totalArms[armId][0] = reward * self.alpha + self.totalArms[armId][0] * (1.0 - self.alpha)
        self.totalArms[armId][1] += 1

    def getTopK(self, numOfSamples):
        self.totalTries += 1
        # normalize the score of all arms: Avg + Confidence
        scores = []

        if self.orderedKeys is None:
            self.orderedKeys = sorted(self.totalArms.keys())

        for key in self.orderedKeys:
            sc = self.totalArms[key][0] + \
                        math.sqrt(0.1*math.log(self.totalTries)/float(self.totalArms[key][1]))

            #self.totalArms[key][2] = sc
            scores.append(sc)
            
        # static UCB, take the top-k
        exploitLen = int(numOfSamples*self.exploitation)
        index = sorted(range(len(scores)), reverse=True, key=lambda k: scores[k])[:exploitLen]
        pickedClients = [self.orderedKeys[x] for x in index]

        # exploration 
        while len(pickedClients) < numOfSamples:
            nextId = self.rng.random.choice(self.orderedKeys)
            if nextId not in pickedClients:
                pickedClients.append(nextId)

        return pickedClients

    def getClientReward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms
