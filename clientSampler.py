import time
from client import Client

class ClientSampler(object):

    def __init__(self):
        self.Clients = {}

    def registerClient(self, hostId, clientId, dis, speed = 1.0):
        uniqueId = str(hostId) + '_' + str(clientId)
        self.Clients[uniqueId] = Client(hostId, clientId, speed, dis)

    def registerSpeed(self, clientId, speed):
        self.Clients[clientId].speed = speed

    def registerDistance(self, clientId, distance):
        self.Clients[clientId].distance = distance

    def getScore(self, clientId):
        return self.Clients[clientId].getScore()

    def nextClientIdToRun(self, hostId):
        return hostId
