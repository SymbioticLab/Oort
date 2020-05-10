class Client(object):

    def __init__(self, hostId, clientId, dis, size, speed):
        self.hostId = hostId
        self.clientId = clientId
        self.compute_speed = speed[0]
        self.bandwidth = speed[1]
        self.distance = dis
        self.size = size
        self.score = dis

    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    def getCompletionTime(self, batch_size, upload_epoch, model_size):
        return (3.0 * batch_size * upload_epoch/float(self.compute_speed) + model_size/float(self.bandwidth))
        #return (5.0 * batch_size * upload_epoch*float(self.compute_speed)/1000. + model_size/float(self.bandwidth))
