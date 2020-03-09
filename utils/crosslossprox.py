from torch.nn import functional as F

class CrossEntropyLossProx(torch.nn._WeightedLoss):

    __constants__ = ['ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(torch.nn.CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.require_gradList = []

    def forward(self, input, target, individual_weight=None, global_weight=None, mu=0):

        init_loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        if global_weight is not None:

            if len(self.require_gradList) == 0:
                for name, param in individual_weight:
                    if param.requires_grad:
                        self.require_gradList.append(name)
                logging.info("====self.require_gradList is {}".format(self.require_gradList))
            else:
                for name in self.require_gradList:
                    init_loss = init_loss + mu * 0.5 * ((individual_weight[name] - global_weight[name]).norm(2) ** 2.)

        return init_loss
