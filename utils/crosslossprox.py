from torch.nn import functional as F
import torch

class CrossEntropyLossProx(torch.nn.Module):

    __constants__ = ['ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.require_gradList = []

    def forward(self, input, target, individual_weight=None, global_weight=None, mu=0):

        init_loss = F.cross_entropy(input, target, reduction='mean')
        surrogateLoss = 0.

        if individual_weight is not None:
            individualList = [param for param in individual_weight]
            globalList = [param for param in global_weight]

            for idx in range(len(individualList)):
                surrogateLoss += ((individualList[idx] - globalList[idx]).norm(2) ** 2.0)

        init_loss = init_loss + mu * 0.5 * surrogateLoss

        return init_loss
