import math
from random import Random
from collections import OrderedDict
import logging

class UCB(object):

    def __init__(self, sample_seed):
        self.totalArms = OrderedDict()
        self.totalTries = 0

        self.exploration = 0.9
        self.decay_factor = 0.99
        self.exploration_min = 0.1
        self.alpha = 0.3

        self.pacer_delta = 1
        self.pacer = -self.pacer_delta

        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()

    def registerArm(self, armId, size, reward):
        # Initiate the score for arms. [score, # of tries]
        if armId not in self.totalArms:
             self.totalArms[armId] = [-1, -1, 0, size]
             self.unexplored.add(armId)

    def registerReward(self, armId, reward, time_stamp):
        # [reward, time stamp]
        self.totalArms[armId][0] = reward # + self.totalArms[armId][0] * (1.0 - self.alpha)
        self.totalArms[armId][1] = time_stamp
        self.totalArms[armId][2] += 1
        self.unexplored.discard(armId)

    def getTopK(self, numOfSamples, cur_time):
        self.totalTries += 1
        # normalize the score of all arms: Avg + Confidence
        scores = []
        numOfExploited = 0

        orderedKeys = list(self.totalArms.keys())
        allloss = []

        moving_reward, staleness = [], []

        for sampledId in orderedKeys:
            if self.totalArms[sampledId][1] != -1:
                moving_reward.append(self.totalArms[sampledId][0])
                staleness.append(cur_time - self.totalArms[sampledId][1])

        max_reward = max(moving_reward)
        min_reward = min(moving_reward)
        range_reward = max_reward - min_reward

        max_staleness = max(staleness)
        min_staleness = min(staleness)
        range_staleness = max(max_staleness - min_staleness, 1)

        for key in orderedKeys:
            # we have played this arm before
            sc = 99999

            if self.totalArms[key][1] != -1:
                sc = (self.totalArms[key][0] - min_reward)/float(range_reward) \
                        - self.alpha*((cur_time-self.totalArms[key][1]) - min_staleness)/float(range_staleness)

                allloss.append(self.totalArms[key][0])
                numOfExploited += 1

            scores.append(sc)

        # static UCB, take the top-k
        self.exploration = max(self.exploration*self.decay_factor, self.exploration_min)
        exploitation = 1.0 - self.exploration

        exploitLen = int(numOfSamples*exploitation)
        self.pacer += self.pacer_delta
        pacer_to = min(self.pacer + exploitLen, len(scores))
        self.pacer = max(0, min(self.pacer, len(scores) - exploitLen))

        index = sorted(range(len(scores)), reverse=False, key=lambda k: scores[k])[self.pacer:pacer_to]
        pickedClients = [orderedKeys[x] for x in index]

        # exploration 
        if len(self.unexplored) > 0:
            _unexplored = list(self.unexplored)
            self.rng.shuffle(_unexplored)
            exploreLen = min(len(_unexplored), numOfSamples - len(pickedClients))

            pickedClients = pickedClients + _unexplored[:exploreLen]

        while len(pickedClients) < numOfSamples:
            nextId = self.rng.choice(orderedKeys)
            if nextId not in pickedClients:
                pickedClients.append(nextId)

        top_k_score = []
        for i in range(min(3, len(pickedClients))):
            clientId = pickedClients[i]
            top_k_score.append([self.totalArms[clientId][0], self.totalArms[clientId][2], self.totalArms[clientId][3], (self.totalArms[clientId][0] - min_reward)/float(range_reward), self.alpha*((cur_time-self.totalArms[clientId][1]) - min_staleness)/float(range_staleness)])
        
        last_exploit = pickedClients[exploitLen-1]
        top_k_score.append([self.totalArms[last_exploit][0], self.totalArms[last_exploit][2], self.totalArms[last_exploit][3], (self.totalArms[last_exploit][0] - min_reward)/float(range_reward), self.alpha*((cur_time-self.totalArms[last_exploit][1]) - min_staleness)/float(range_staleness)])

        logging.info("====At time {}, UCB exploited {}, un-explored {}, pacer {} to {}, top-k score is {}"
            .format(cur_time, numOfExploited, len(self.totalArms) - numOfExploited, self.pacer, pacer_to, top_k_score))
        logging.info("====At time {}, all rewards are {}".format(cur_time, allloss))

        return pickedClients

    def getClientReward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms
