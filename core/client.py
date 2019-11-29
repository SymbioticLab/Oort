class Client(object):

    def __init__(self, hostId, clientId, dis, speed, size):
        self.hostId = hostId
        self.clientId = clientId
        self.speed = speed
        self.distance = dis
        self.size = size
        #self.lastPiggBack = time.time()

    def getScore(self):
        return speed * distance
