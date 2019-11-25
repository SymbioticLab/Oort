class Client(object):

    def __init__(self, hostId, clientId, dis, speed):
        self.hostId = hostId
        self.clientId = clientId
        self.speed = speed
        self.distance = dis
        #self.lastPiggBack = time.time()

    def getScore(self):
        return speed * distance
