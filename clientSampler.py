from core.ucb import UCB
from core.client import Client
import math
from random import Random
import logging
from core.argParser import args

class ClientSampler(object):

    def __init__(self, mode, score, filter=0, sample_seed=233):
        self.Clients = {}
        self.clientOnHosts = {}
        self.mode = mode
        self.score = score
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more

        self.ucbSampler = UCB(sample_seed=sample_seed, score_mode=score, args=args) if self.mode == "bandit" else None
        self.feasibleClients = []
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.count = 0
        self.feasible_samples = 0

    def registerClient(self, hostId, clientId, dis, size, speed=[1.0, 1.0], duration=1):

        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId] = Client(hostId, clientId, dis, size, speed)

        if size >= self.filter_less and size <= self.filter_more:
            self.feasibleClients.append(clientId)

            self.feasible_samples += size

            if self.mode == "bandit":
                self.ucbSampler.registerArm(clientId, reward=min(size, args.upload_epoch*args.batch_size), size=size, duration=duration)

    def getAllClients(self):
        return self.feasibleClients

    def getClient(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)]

    def registerDuration(self, clientId, batch_size, upload_epoch, model_size):
        if self.mode == "bandit":
            roundDuration = self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                    batch_size=batch_size, upload_epoch=upload_epoch, model_size=model_size
            )
            self.ucbSampler.registerDuration(clientId, roundDuration)

    def getCompletionTime(self, clientId, batch_size, upload_epoch, model_size):
        return self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                batch_size=batch_size, upload_epoch=upload_epoch, model_size=model_size
            )
        
    def registerSpeed(self, hostId, clientId, speed):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].speed = speed

    def registerScore(self, clientId, reward, auxi=1.0, time_stamp=0, duration=1., success=True):
        # currently, we only use distance as reward
        if self.mode == "bandit":
            self.ucbSampler.registerReward(clientId, reward, auxi=auxi, time_stamp=time_stamp, duration=duration, success=success)

        self.registerClientScore(clientId, reward)
    
    def registerClientScore(self, clientId, reward):
        self.Clients[self.getUniqueId(0, clientId)].registerReward(reward)

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
        init_id = hostId - 1
        lenPossible = len(self.feasibleClients)

        while True:
            clientId = str(self.feasibleClients[init_id])
            csize = self.Clients[clientId].size
            if csize >= self.filter_less and csize <= self.filter_more:
                return int(clientId)

            init_id = max(0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))

        return init_id

    def getUniqueId(self, hostId, clientId):
        return str(clientId)
        #return (str(hostId) + '_' + str(clientId))

    def clientSampler(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

    def clientOnHost(self, clientIds, hostId):
        self.clientOnHosts[hostId] = clientIds

    def getCurrentClientIds(self, hostId):
        return self.clientOnHosts[hostId]

    def getClientLenOnHost(self, hostId):
        return len(self.clientOnHosts[hostId])

    def getClientSize(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

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
            return self.ucbSampler.getTopK(numOfClients, cur_time=cur_time)
        else:
            self.rng.shuffle(self.feasibleClients)
            return self.feasibleClients[:numOfClients]

    def getAllMetrics(self):
        if self.mode == "bandit":
            return self.ucbSampler.getAllMetrics()
        return {}

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients), 'total_length': self.feasible_samples}

    def getClientReward(self, clientId):
        return self.ucbSampler.getClientReward(clientId)
