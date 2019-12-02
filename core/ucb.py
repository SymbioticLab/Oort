import math
import numpy as np2

class UCB(object):

    def __init__(self):
        self.totalArms = {}
        self.totalTries = 0
        self.alpha = 0.8
        np2.random.seed(123)

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
        clientIds = []

        for key in self.totalArms.keys():
            sc = self.totalArms[key][0] #+ \
                        #math.sqrt(0.1*math.log(self.totalTries)/float(self.totalArms[key][1]))
            scores.append(sc)
            clientIds.append(key)

        # static UCB
        #index = np2.array(scores).argsort()[-numOfSamples:][::-1] + 1
        scores = np2.array(scores)/float(sum(scores))

        for i, clientId in enumerate(clientIds):
            self.totalArms[clientId][2] = scores[i]

        index = np2.random.choice(clientIds, size=numOfSamples, p = scores.ravel(), replace=False)

        return index

    def getClientReward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms
