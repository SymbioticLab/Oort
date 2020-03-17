from core.ucb import UCB
from core.client import Client
import math
from random import Random
import logging

class ClientSampler(object):

    def __init__(self, mode, score, filter=0, sample_seed=233):
        self.Clients = {}
        self.clientOnHosts = {}
        self.mode = mode
        self.score = score
        self.filter = filter

        self.ucbSampler = UCB(sample_seed=sample_seed) if self.mode == "bandit" else None
        self.feasibleClients = []
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.count = 0

    def registerClient(self, hostId, clientId, dis, size, speed = 1.0):

        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId] = Client(hostId, clientId, dis, size, speed)

        if size >= self.filter:
            self.feasibleClients.append(clientId)

            if self.mode == "bandit":
                #if self.score == "loss":
                self.ucbSampler.registerArm(clientId, reward=10.0 - dis, size=size)
                #else:
                #    self.ucbSampler.registerArm(clientId, reward=1.0 - dis, size=size)

    def registerSpeed(self, hostId, clientId, speed):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].speed = speed

    def registerScore(self, clientId, reward, time_stamp=0):
        # currently, we only use distance as reward
        if self.mode == "bandit":
            self.ucbSampler.registerReward(clientId, reward, time_stamp)

    def getScore(self, hostId, clientId):
        uniqueId = self.getUniqueId(hostId, clientId)
        return self.Clients[uniqueId].getScore()

    def getClientsInfo(self):
        clientInfo = {}
        for i, clientId in enumerate(self.Clients.keys()):
            client = self.Clients[clientId]
            clientInfo[client.clientId] = client.distance
        return clientInfo

    def nextClientIdToRun(self, hostId):
        init_id = hostId
        lenPossible = len(self.feasibleClients)

        while True:
            clientId = str(self.feasibleClients[init_id])
            if self.Clients[clientId].size >= self.filter:
                return int(clientId)

            init_id = max(0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))

        return init_id

    def getUniqueId(self, hostId, clientId):
        return str(clientId)
        #return (str(hostId) + '_' + str(clientId))

    def clientOnHost(self, clientIds, hostId):
        self.clientOnHosts[hostId] = clientIds

    def getCurrentClientIds(self, hostId):
        return self.clientOnHosts[hostId]

    def getClientLenOnHost(self, hostId):
        return len(self.clientOnHosts[hostId])

    def getSampleRatio(self, clientId, hostId, even=False):
        totalSampleInTraining = 0.

        if not even:
            for key in self.clientOnHosts.keys():
                for client in self.clientOnHosts[key]:
                    uniqueId = self.getUniqueId(key, client)
                    totalSampleInTraining += self.Clients[uniqueId].size

            #1./len(self.clientOnHosts.keys())
            return float(self.Clients[self.getUniqueId(hostId, clientId)].size)/float(totalSampleInTraining)
        else:
            for key in self.clientOnHosts.keys():
                totalSampleInTraining += len(self.clientOnHosts[key])

            return 1./totalSampleInTraining

    def resampleClients(self, numOfClients, cur_time=0):
        self.count += 1

        if self.mode == "bandit" and self.count > 1:
            if self.score == "norm":
                return self.ucbSampler.getTopKByNorm(numOfClients, cur_time=cur_time)
            else:
                return self.ucbSampler.getTopK(numOfClients, cur_time=cur_time)
        else:
            self.rng.shuffle(self.feasibleClients)
            return self.feasibleClients[:numOfClients]

    def getAllMetrics(self):
        if self.mode == "bandit":
            return self.ucbSampler.getAllMetrics()
        return {}

    def getClientReward(self, clientId):
        return self.ucbSampler.getClientReward(clientId)
