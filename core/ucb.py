import math
from random import Random
from collections import OrderedDict
import logging

class UCB(object):

    def __init__(self, sample_seed):
        self.totalArms = OrderedDict()
        self.totalTries = 0
        self.alpha = 0.8

        self.exploration = 0.2
        self.exploitation = 1.0 - self.exploration

        self.rng = Random()
        self.rng.seed(sample_seed)

        #self.orderedKeys = None

    def registerArm(self, armId, reward):
        # Initiate the score for arms. [score, # of tries]
        if armId not in self.totalArms:
             self.totalArms[armId] = [-1, -1, 0]

    def registerReward(self, armId, reward, time_stamp):
        # [reward, time stamp]
        self.totalArms[armId][0] = reward # + self.totalArms[armId][0] * (1.0 - self.alpha)
        self.totalArms[armId][1] = time_stamp
        self.totalArms[armId][2] += 1

    def getTopK(self, numOfSamples):
        self.totalTries += 1
        # normalize the score of all arms: Avg + Confidence
        scores = []

        #if self.orderedKeys is None:
        orderedKeys = list(self.totalArms.keys())

        for key in orderedKeys:
            # we have played this arm before
            sc = -1.0
            if self.totalArms[key][1] != -1:
                sc = self.totalArms[key][0] + \
                            math.sqrt(0.1*math.log(self.totalTries)/float(self.totalArms[key][1]))

            scores.append(sc)
            
        # static UCB, take the top-k
        exploitLen = int(numOfSamples*self.exploitation)
        index = sorted(range(len(scores)), reverse=True, key=lambda k: scores[k])[:exploitLen]
        pickedClients = [orderedKeys[x] for x in index]

        # exploration 
        while len(pickedClients) < numOfSamples:
            nextId = self.rng.choice(orderedKeys)
            if nextId not in pickedClients:
                pickedClients.append(nextId)

        logging.info("====For {} times, UCB works to pick {} from {}".format(self.totalTries, exploitLen, numOfSamples))
        return pickedClients

    def getClientReward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms
