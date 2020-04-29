import math
from random import Random
from collections import OrderedDict
import logging, pickle
import numpy as np2

class UCB(object):

    def __init__(self, sample_seed, score_mode, args):
        self.totalArms = OrderedDict()
        self.training_round = 0

        self.exploration = 0.9
        self.decay_factor = 0.98
        self.exploration_min = 0.2
        self.alpha = 0.3

        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        self.score_mode = score_mode
        self.args = args
        self.round_threshold = args.round_threshold
        self.round_prefer_duration = 99999999999
        self.last_util_record = 0

        self.sample_window = self.args.sample_window
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []
        self.successfullClients = set()

        np2.random.seed(sample_seed)

    def registerArm(self, armId, size, reward, duration):
        # Initiate the score for arms. [score, time_stamp, # of trials, size of client, auxi, duration]
        if armId not in self.totalArms:
             self.totalArms[armId] = [0., 0., 0., float(size), 0., duration]
             self.unexplored.add(armId)

    def registerDuration(self, armId, duration):
        if armId in self.totalArms:
            self.totalArms[armId][5] = duration

    def calculateSumUtil(self, clientList):
        cnt = 0
        cntUtil = 0.
        for client in clientList:
            if self.totalArms[client][1] == self.training_round - 1 and client in self.successfullClients:
                cnt += 1
                cntUtil += self.totalArms[client][0]

        if cntUtil > 0:
            return cntUtil/cnt

        return 0

    def pacer(self):
        # summarize utility in last epoch
        lastExplorationUtil = self.calculateSumUtil(self.exploreClients)
        lastExploitationUtil = self.calculateSumUtil(self.exploitClients)

        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)

        self.successfullClients = set()

        if self.training_round >= 2 * self.args.pacer_step and self.training_round % self.args.pacer_step == 0:
            # if we notice the overall utility does not increase, we would expand by relaxing system constraints
            
            #utilPenultimateRounds = sum(self.exploitUtilHistory[-3*self.args.pacer_step:-2*self.args.pacer_step])
            utilLastPacerRounds = sum(self.exploitUtilHistory[last_util_record:last_util_record+self.args.pacer_step])
            utilCurrentPacerRounds = sum(self.exploitUtilHistory[-self.args.pacer_step:])

            if utilCurrentPacerRounds <= utilLastPacerRounds * 0.99:
                self.round_threshold = min(100., self.round_threshold + self.args.pacer_delta)
                self.last_util_record = self.training_round -self.args.pacer_step
            elif utilCurrentPacerRounds >= utilLastPacerRounds * 1.15:
                self.round_threshold = max(self.args.pacer_delta, self.round_threshold - self.args.pacer_delta)
                self.last_util_record = self.training_round -self.args.pacer_step

            logging.info("====utilLastPacerRounds {}, utilCurrentPacerRounds {}".format(utilLastPacerRounds, utilCurrentPacerRounds))

        logging.info("====Pacer {}: lastExploitationUtil {}, lastExplorationUtil {}, last_util_record {}".
                        format(self.training_round, lastExploitationUtil, lastExplorationUtil, self.last_util_record))

    def registerReward(self, armId, reward, auxi, time_stamp, duration, success=True):
        # [reward, time stamp]
        self.totalArms[armId][0] = reward
        self.totalArms[armId][1] = time_stamp
        self.totalArms[armId][2] += 1
        self.totalArms[armId][4] = auxi
        self.totalArms[armId][5] = duration 

        self.unexplored.discard(armId)

        if success:
            self.successfullClients.add(armId)

    def getExpectation(self):
        sum_reward = 0.
        sum_count = 0.

        for arm in self.totalArms:
            if self.totalArms[arm][0] > 0:
                sum_count += 1
                sum_reward += self.totalArms[arm][4]

        return sum_reward/sum_count

    def getTopK(self, numOfSamples, cur_time):
        self.training_round = cur_time

        self.pacer()
        # normalize the score of all arms: Avg + Confidence
        scores = {}
        numOfExploited = 0
        exploreLen = 0

        orderedKeys = list(self.totalArms.keys())
        if self.round_threshold < 100:
            sortedDuration = sorted([self.totalArms[key][5] for key in orderedKeys])
            self.round_prefer_duration = sortedDuration[min(int(len(sortedDuration) * self.round_threshold/100.), len(sortedDuration)-1)]

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
            if self.totalArms[key][2] > 0:
                creward = self.totalArms[key][0]
                numOfExploited += 1

                sc = (creward - min_reward)/float(range_reward) \
                    + math.sqrt(0.1*math.log(cur_time)/self.totalArms[key][1]) # temporal-uncertainty

                    #self.alpha*((cur_time-self.totalArms[key][1]) - min_staleness)/float(range_staleness)

                clientDuration = self.totalArms[key][5]
                if clientDuration > self.round_prefer_duration:
                    sc *= ((float(self.round_prefer_duration)/clientDuration) ** self.args.round_penalty)

                if self.totalArms[key][1] == cur_time - 1:
                    allloss[key] = sc

                scores[key] = sc

        clientLakes = list(scores.keys())
        self.exploration = max(self.exploration*self.decay_factor, self.exploration_min)
        exploitLen = min(int(numOfSamples*(1.0 - self.exploration)), len(clientLakes))

        # take the top-k, and then sample by probability
        pickedClients = sorted(scores, key=scores.get, reverse=True)[:int(self.sample_window*exploitLen)]
        totalSc = float(sum([scores[key] for key in pickedClients]))
        pickedClients = list(np2.random.choice(pickedClients, exploitLen, p=[scores[key]/totalSc for key in pickedClients], replace=False))
        self.exploitClients = pickedClients

        # exploration 
        if len(self.unexplored) > 0:
            _unexplored = list(self.unexplored)

            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = self.totalArms[cl][3]
                clientDuration = self.totalArms[cl][5]

                if clientDuration > self.round_prefer_duration:
                    init_reward[cl] *= ((float(self.round_prefer_duration)/clientDuration) ** self.args.round_penalty)

            # prioritize w/ some rewards (i.e., size)
            exploreLen = min(len(_unexplored), numOfSamples - len(pickedClients))
            pickedUnexploredClients = sorted(init_reward, key=init_reward.get, reverse=True)[:min(int(self.sample_window*exploreLen), len(init_reward))]

            unexploredSc = float(sum([init_reward[key] for key in pickedUnexploredClients]))
            #sorted(_unexplored, reverse=True, key=lambda k:init_reward[k])
            #self.rng.shuffle(_unexplored)
            pickedUnexplored = list(np2.random.choice(pickedUnexploredClients, exploreLen, 
                                p=[init_reward[key]/unexploredSc for key in pickedUnexploredClients], replace=False))

            self.exploreClients = pickedUnexplored
            pickedClients = pickedClients + pickedUnexplored
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
            _staleness = math.sqrt(0.1*math.log(cur_time)/self.totalArms[clientId][1]) #self.alpha*((cur_time-self.totalArms[clientId][1]) - min_staleness)/float(range_staleness)
            top_k_score.append(self.totalArms[clientId] + [_score, _staleness])

        last_exploit = pickedClients[exploitLen-1]
        top_k_score.append(self.totalArms[last_exploit] + \
            [(self.totalArms[last_exploit][0] - min_reward)/float(range_reward), self.alpha*((cur_time-self.totalArms[last_exploit][1]) - min_staleness)/float(range_staleness)])

        logging.info("====At time {}, UCB exploited {}, exploreLen {}, un-explored {}, indeed un-explored {}, exploration {}, round_threshold {}, top-k score is {}"
            .format(cur_time, numOfExploited, exploreLen, len(self.totalArms) - numOfExploited, len(self.unexplored), self.exploration, self.round_threshold, top_k_score))
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
        _max = max(aList)*0.95
        _min = min(aList)*0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList)/float(len(aList))

        return float(_max), float(_min), float(_range), float(_avg)
