import time
from client import Client

class ClientSampler(object):

    def __init__(self):
        self.Clients = {}

    def registerClient(self, hostId, clientId, dis, speed = 1.0):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId] = Client(hostId, clientId, speed, dis)

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
        return self.Clients

    def nextClientIdToRun(self, hostId):
        return hostId

    def getUniqueId(self, hostId, clientId):
        return (str(hostId) + '_' + str(clientId))

