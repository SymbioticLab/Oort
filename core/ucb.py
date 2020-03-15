import math
from random import Random
from collections import OrderedDict
import logging

class UCB(object):

    def __init__(self, sample_seed):
        self.totalArms = OrderedDict()
        self.totalTries = 0
        self.alpha = 0.8

        self.exploration = 0.8
        self.exploitation = 1.0 - self.exploration
        self.decay_factor = 0.99

        self.rng = Random()
        self.rng.seed(sample_seed)

        #self.orderedKeys = None

    def registerArm(self, armId, size, reward):
        # Initiate the score for arms. [score, # of tries]
        if armId not in self.totalArms:
             self.totalArms[armId] = [-1, -1, 0, size]

    def registerReward(self, armId, reward, time_stamp):
        # [reward, time stamp]
        self.totalArms[armId][0] = reward # + self.totalArms[armId][0] * (1.0 - self.alpha)
        self.totalArms[armId][1] = time_stamp
        self.totalArms[armId][2] += 1

    def getTopK(self, numOfSamples, cur_time):
        self.totalTries += 1
        # normalize the score of all arms: Avg + Confidence
        scores = []
        numOfExploited = 0

        #if self.orderedKeys is None:
        orderedKeys = list(self.totalArms.keys())

        for key in orderedKeys:
            # we have played this arm before
            sc = -1.0
            if self.totalArms[key][1] != -1:
                numOfExploited += 1
                sc = self.totalArms[key][0] + \
                            math.sqrt(1.5*math.log(cur_time)/float(self.totalArms[key][1]))

            scores.append(sc)
            
        # static UCB, take the top-k
        self.exploration = max(self.exploration*self.decay_factor, 0.2)
        self.exploitation = 1.0 - self.exploration

        exploitLen = int(numOfSamples*self.exploitation)
        index = sorted(range(len(scores)), reverse=True, key=lambda k: scores[k])[:exploitLen]
        pickedClients = [orderedKeys[x] for x in index]

        # exploration 
        while len(pickedClients) < numOfSamples:
            nextId = self.rng.choice(orderedKeys)
            if nextId not in pickedClients:
                pickedClients.append(nextId)

        top_k_score = []
        for i in range(min(3, len(pickedClients))):
            clientId = pickedClients[i]
            top_k_score.append([self.totalArms[clientId][0], self.totalArms[clientId][2], self.totalArms[clientId][3], math.sqrt(1.5*math.log(cur_time)/float(self.totalArms[clientId][1]))])
        last_exploit = pickedClients[exploitLen-1]
        top_k_score.append([self.totalArms[last_exploit][0], self.totalArms[last_exploit][2], self.totalArms[last_exploit][3], math.sqrt(1.5*math.log(cur_time)/float(self.totalArms[last_exploit][1]))])

        logging.info("====At time {}, UCB exploited {}, un-explored {}, top-k score is {}".format(cur_time, numOfExploited, len(self.totalArms) - numOfExploited, top_k_score))
        return pickedClients

    def getClientReward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms
