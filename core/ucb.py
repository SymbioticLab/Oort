import math
import numpy as np2

class UCB(object):

    def __init__(self):
        self.totalArms = {}
        self.totalTries = 0
        np2.random.seed(123)

    def registerArm(self, armId):
        # Initiate the score for arms. [score, # of tries]
        if armId not in self.totalArms:
            self.totalArms[armId] = [0., 1]

    def registerReward(self, armId, reward):
        self.totalArms[armId][0] += reward
        self.totalArms[armId][1] += 1

    def getTopK(self, numOfSamples):
        self.totalTries += 1
        # normalize the score of all arms: Avg + Confidence
        scores = []

        for key in self.totalArms.keys():
            sc = self.totalArms[key][0]/float(self.totalArms[key][1]) + 
                        math.sqrt(2.0*math.log(self.totalTries)/float(self.totalArms[armId][1]))

            scores.append(sc)

        scores = scores/float(sum(scores))
        index = np2.random.choice([i for i in range(1, len(scores) + 1)], p = np2.array(scores).ravel(), replace=False)

        return index

    def getAllMetrics(self):
        return self.totalArms
