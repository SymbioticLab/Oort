class Client(object):

    def __init__(self, hostId, clientId, dis, size, speed):
        self.hostId = hostId
        self.clientId = clientId
        self.speed = speed
        self.distance = dis
        self.size = size
        self.score = dis
        #self.lastPiggBack = time.time()

    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward
