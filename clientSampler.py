import time

class Client(object):

    def __init__(self, id, speed, dis = 0):
        self.id = id
        self.speed = speed
        self.distance = dis
        self.lastPiggBack = time.time()

    def getScore(self):
        return speed * distance

class clientSampler(object):

    def __init__(self, numOfClients):
        self.Clients = []

        for clientId in range(numOfClients):
            self.Clients.append(Client(clientId, 1.0))

    def registerSpeed(clientId, speed):
        self.Clients[clientId].speed = speed
        #self.Clients[clientId].lastPiggBack = time.time()

    def registerDistance(clientId, distance):
        self.Clients[clientId].distance = distance

    def getScore(self, clientId):
        return self.Clients[clientId].getScore()


