import math
from random import Random
from collections import OrderedDict
import logging, pickle
import numpy as np2

class UCB(object):

    def __init__(self, sample_seed, score_mode, args):
        self.totalArms = OrderedDict()
        self.numOfTrials = 0

        self.exploration = 0.9
        self.decay_factor = 0.98
        self.exploration_min = 0.2
        self.alpha = 0.3

        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        self.score_mode = score_mode
        self.args = args

        self.sample_window = self.args.sample_window
        np2.random.seed(sample_seed)

    def registerArm(self, armId, size, reward, duration):
        # Initiate the score for arms. [score, time_stamp, # of trials, size of client, auxi, duration]
        if armId not in self.totalArms:
             self.totalArms[armId] = [0., 0., 0., float(size), 0., duration]
             self.unexplored.add(armId)

    def registerDuration(self, armId, duration):
        if armId in self.totalArms:
            self.totalArms[armId][5] = duration

    def registerReward(self, armId, reward, auxi, time_stamp, duration):
        # [reward, time stamp]
        self.totalArms[armId][0] = reward
        self.totalArms[armId][1] = time_stamp
        self.totalArms[armId][2] += 1
        self.totalArms[armId][4] = auxi
        self.totalArms[armId][5] = duration 

        self.unexplored.discard(armId)

    def getExpectation(self):
        sum_reward = 0.
        sum_count = 0.

        for arm in self.totalArms:
            if self.totalArms[arm][0] > 0:
                sum_count += 1
                sum_reward += self.totalArms[arm][4]

        return sum_reward/sum_count

    def getTopK(self, numOfSamples, cur_time):
        self.numOfTrials += 1
        # normalize the score of all arms: Avg + Confidence
        scores = {}
        numOfExploited = 0
        exploreLen = 0

        orderedKeys = list(self.totalArms.keys())

        moving_reward, staleness, allloss = [], [], {}
        expectation_reward = self.getExpectation()

        for sampledId in orderedKeys:
            if self.totalArms[sampledId][0] > 0:
                creward = self.totalArms[sampledId][0]
                moving_reward.append(creward)
                staleness.append(cur_time - self.totalArms[sampledId][1])

        max_reward, min_reward, range_reward, avg_reward = self.get_norm(moving_reward)
        max_staleness, min_staleness, range_staleness, avg_staleness = self.get_norm(staleness, thres=1)
        

        for key in orderedKeys:
            # we have played this arm before
            if self.totalArms[key][0] > 0:
                creward = self.totalArms[key][0]
                numOfExploited += 1

                sc = (creward - min_reward)/float(range_reward) \
                    + self.alpha*((cur_time-self.totalArms[key][1]) - min_staleness)/float(range_staleness)

                clientDuration = self.totalArms[key][5]
                if clientDuration > self.args.round_threshold:
                    sc *= ((float(self.args.round_threshold)/clientDuration) ** self.args.round_penalty)

                if self.totalArms[key][1] == cur_time - 1:
                    allloss[key] = sc

                scores[key] = sc

        clientLakes = list(scores.keys())
        self.exploration = max(self.exploration*self.decay_factor, self.exploration_min)
        exploitLen = min(int(numOfSamples*(1.0 - self.exploration)), len(clientLakes))

        # static UCB, take the top-k
        pickedClients = sorted(scores, key=scores.get, reverse=True)[:int(self.sample_window*exploitLen)]
        totalSc = float(sum([scores[key] for key in pickedClients]))
        pickedClients = list(np2.random.choice(pickedClients, exploitLen, p=[scores[key]/totalSc for key in pickedClients], replace=False))

        # dynamic UCB, pick by probablity
        #totalSc = float(sum([scores[key] for key in clientLakes]))
        #pickedClients = list(np2.random.choice(clientLakes, exploitLen, p=[scores[key]/totalSc for key in clientLakes], replace=False))

        # exploration 
        if len(self.unexplored) > 0:
            _unexplored = list(self.unexplored)

            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = self.totalArms[cl][3]
                clientDuration = self.totalArms[cl][5]

                if clientDuration > self.args.round_threshold:
                    init_reward[cl] *= ((float(self.args.round_threshold)/clientDuration) ** self.args.round_penalty)

            # prioritize w/ some rewards (i.e., size)
            unexplored_by_rewards = sorted(_unexplored, reverse=True, key=lambda k:init_reward[k])
            #self.rng.shuffle(_unexplored)
            exploreLen = min(len(_unexplored), numOfSamples - len(pickedClients))
            pickedClients = pickedClients + unexplored_by_rewards[:exploreLen]
        else:
            # no clients left for exploration
            self.exploration_min = 0.
            self.exploration = 0.

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

        logging.info("====At time {}, UCB exploited {}, exploreLen {}, un-explored {}, top-k score is {}"
            .format(cur_time, numOfExploited, exploreLen, len(self.totalArms) - numOfExploited, top_k_score))
        logging.info("====At time {}, all rewards are {}".format(cur_time, allloss))

        return pickedClients

    def getClientReward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms

    def get_norm(self, aList, thres=0):
        aList = sorted(aList)
        # _95th = aList[int(len(aList)*0.95)]
        # _5th = aList[int(len(aList)*0.05)]

        # return _95th, _5th, max((_95th - _5th), thres)
        _max = max(aList)
        _min = min(aList)*0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList)/float(len(aList))

        return float(_max), float(_min), float(_range), float(_avg)
