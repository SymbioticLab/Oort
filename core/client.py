class Client(object):

    def __init__(self, hostId, clientId, dis, size, speed):
        self.hostId = hostId
        self.clientId = clientId
        self.speed = speed
        self.distance = dis
        self.size = size
        #self.lastPiggBack = time.time()

    def getScore(self):
        return self.distance#self.speed * self.distance
