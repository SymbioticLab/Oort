from core.ucb import UCB
from core.client import Client
import random

class ClientSampler(object):

    def __init__(self, mode):
        self.Clients = {}
        self.clientOnHosts = {}
        self.mode = mode
        random.seed(123)

        self.ucbSampler = UCB() if self.mode == "bandit" else None

    def registerClient(self, hostId, clientId, dis, size, speed = 1.0):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId] = Client(hostId, clientId, dis, speed, size)

        if self.mode == "bandit":
            self.ucbSampler.registerArm(clientId, dis)

    def registerSpeed(self, hostId, clientId, speed):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].speed = speed

    def registerScore(self, clientId, reward):
        # currently, we only use distance as reward
        if self.mode == "bandit":
            self.ucbSampler.registerReward(clientId, reward)

    def getScore(self, hostId, clientId):
        uniqueId = self.getUniqueId(hostId, clientId)
        return self.Clients[uniqueId].getScore()

    def getClientsInfo(self):
        clientInfo = {}
        for i, clientId in enumerate(self.Clients.keys()):
            client = self.Clients[clientId]
            clientInfo[i] = {'Id': client.clientId, 'Distance': client.distance}
        return clientInfo

    def nextClientIdToRun(self, hostId):
        return hostId

    def getUniqueId(self, hostId, clientId):
        return (str(hostId) + '_' + str(clientId))

    def clientOnHost(self, clientId, hostId):
        self.clientOnHosts[hostId] = clientId

    def getSampleRatio(self, clientId, hostId):
        totalSampleInTraining = 0.

        for key in self.clientOnHosts:
            uniqueId = self.getUniqueId(key, self.clientOnHosts[key])
            totalSampleInTraining += self.Clients[uniqueId].size

        return (float(self.Clients[self.getUniqueId(hostId, clientId)].size)/totalSampleInTraining)

    def resampleClients(self, numOfClients, totalClients):
        if self.mode == "bandit":
            return self.ucbSampler.getTopK(numOfClients)
        else:
            return random.sample(range(1, totalClients+1), numOfClients)

    def getCurrentClientId(self, hostId):
        return self.clientOnHosts[hostId]

    def getAllMetrics(self):
        if self.mode == "bandit":
            return self.ucbSampler.getAllMetrics()
        return {}

