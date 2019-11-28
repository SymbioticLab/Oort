import random
from client import Client

random.seed(123)

class ClientSampler(object):

    def __init__(self):
        self.Clients = {}
        self.clientOnHosts = {}

    def registerClient(self, hostId, clientId, dis, speed = 1.0):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId] = Client(hostId, clientId, dis, speed)

    def registerSpeed(self, hostId, clientId, speed):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].speed = speed

    def registerDistance(self, hostId, clientId, distance):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].distance = distance

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

    def resampleClients(self, numOfClients, totalClients, mode):
        if mode == "random":
            return random.sample(range(1, totalClients+1), numOfClients)

    def getCurrentClientId(self, hostId):
        return self.clientOnHosts[hostId]




