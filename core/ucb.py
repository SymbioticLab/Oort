import math
from random import Random
from collections import OrderedDict
import logging

class UCB(object):

    def __init__(self, sample_seed, score_mode):
        self.totalArms = OrderedDict()
        self.numOfTrials = 0

        self.exploration = 0.9
        self.decay_factor = 0.99
        self.exploration_min = 0.1
        self.alpha = 0.3

        self.pacer_delta = 0.5 if score_mode == "loss" else 0
        self.pacer = -self.pacer_delta

        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        self.score_mode = score_mode

    def registerArm(self, armId, size, reward):
        # Initiate the score for arms. [score, time_stamp, # of trials, size of client]
        if armId not in self.totalArms:
             self.totalArms[armId] = [-1, -1, 0, size]
             self.unexplored.add(armId)

    def registerReward(self, armId, reward, time_stamp):
        # [reward, time stamp]
        self.totalArms[armId][0] = reward
        self.totalArms[armId][1] = time_stamp
        self.totalArms[armId][2] += 1
        self.unexplored.discard(armId)

    def getTopK(self, numOfSamples, cur_time):
        self.numOfTrials += 1
        # normalize the score of all arms: Avg + Confidence
        scores = {}
        numOfExploited = 0
        orderedKeys = list(self.totalArms.keys())

        moving_reward, staleness, allloss = [], [], {}

        for sampledId in orderedKeys:
            if self.totalArms[sampledId][1] != -1:
                moving_reward.append(self.totalArms[sampledId][0])
                staleness.append(cur_time - self.totalArms[sampledId][1])

        max_reward, min_reward, range_reward = self.get_norm(moving_reward)
        max_staleness, min_staleness, range_staleness = self.get_norm(staleness, thres=1)

        for key in orderedKeys:
            # we have played this arm before
            if self.totalArms[key][1] != -1:
                if self.score_mode == "loss":
                    sc = (self.totalArms[key][0] - min_reward)/float(range_reward) \
                        - self.alpha*((cur_time-self.totalArms[key][1]) - min_staleness)/float(range_staleness)
                else:
                    sc = (self.totalArms[key][0] - min_reward)/float(range_reward) \
                        + self.alpha*((cur_time-self.totalArms[key][1]) - min_staleness)/float(range_staleness)

                if self.totalArms[key][1] == cur_time - 1:
                    allloss[key] = self.totalArms[key][0]
                numOfExploited += 1
                scores[key] = sc

        # static UCB, take the top-k
        self.exploration = max(self.exploration*self.decay_factor, self.exploration_min)
        exploitLen = int(numOfSamples*(1.0 - self.exploration))

        self.pacer += self.pacer_delta
        self.pacer = max(0, min(self.pacer, len(scores) - exploitLen))

        pacer_from = int(self.pacer)
        pacer_to = min(pacer_from + exploitLen, len(scores))

        isReverse = False if self.score_mode == "loss" else True
        pickedClients = sorted(scores, key=scores.get, reverse=isReverse)[pacer_from:pacer_to]

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
            _score = (self.totalArms[clientId][0] - min_reward)/float(range_reward)
            _staleness = self.alpha*((cur_time-self.totalArms[clientId][1]) - min_staleness)/float(range_staleness)
            top_k_score.append(self.totalArms[clientId] + [_score, _staleness])
        
        last_exploit = pickedClients[exploitLen-1]
        top_k_score.append(self.totalArms[last_exploit] + \
            [(self.totalArms[last_exploit][0] - min_reward)/float(range_reward), self.alpha*((cur_time-self.totalArms[last_exploit][1]) - min_staleness)/float(range_staleness)])

        logging.info("====At time {}, UCB exploited {}, un-explored {}, pacer {} to {}, top-k score is {}"
            .format(cur_time, numOfExploited, len(self.totalArms) - numOfExploited, self.pacer, pacer_to, top_k_score))
        logging.info("====At time {}, all rewards are {}".format(cur_time, allloss))

        return pickedClients

    def getClientReward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms

    def get_norm(self, aList, thres=0):
        aList = sorted(aList)
        _95th = aList[int(len(aList)*0.95)]
        _5th = aList[int(len(aList)*0.05)]

        return _95th, _5th, max((_95th - _5th), thres)
        # _max = max(aList)
        # _min = min(aList)
        # _range = max(_max - _min, thres)

        # return _max, _min, _range

