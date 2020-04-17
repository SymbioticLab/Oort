# -*- coding: utf-8 -*-

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import logging
from core.argParser import args
from utils.nlp import mask_tokens

class MySGD(optim.SGD):

    def __init__(self, params, lr=0.01, momentum=0.0,
                 dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # print('Previous: {}, lr: {}, grad: {}'.format(p.data, group['lr'], d_p))
                p.data.add_(-group['lr'], d_p)
                # print('Now: {}'.format(p.data))

        return loss

    def get_delta_w(self, nestedLr=0.01):
        delta_ws = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if nestedLr == 0.01:
                    delta_ws.append(group['lr'] * d_p)
                else:
                    delta_ws.append(nestedLr * d_p)

        return delta_ws

def test_model(rank, model, test_data, criterion=nn.NLLLoss(), tokenizer=None):
    test_loss = 0
    correct = 0
    top_5 = 0

    correct2 = 0
    test_len = 0
    perplexity_loss = 0.

    model.eval()
    for data, target in test_data:
        if args.task == 'tag':
            data, target = Variable(data).cuda(), Variable(target).cuda()

            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.data.item()  # Variable.data
            # acc = accuracy(output, target, topk=(1, 5))

            # correct += acc[0].item()
            # top_5 += acc[1].item()

        elif args.task != 'nlp':
            data, target = Variable(data).cuda(), Variable(target).cuda()

            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.data.item()  # Variable.data
            acc = accuracy(output, target, topk=(1, 5))

            correct += acc[0].item()
            top_5 += acc[1].item()

        else:
            data, target = mask_tokens(data, tokenizer, args) if args.mlm else (data, data)
            data, target = Variable(data).cuda(), Variable(target).cuda()
            
            outputs = model(data, masked_lm_labels=target) if args.mlm else model(data, labels=target)
            test_loss += outputs[0].item()
            perplexity_loss += outputs[0].mean().item()

        test_len += len(target)
        
    # loss function averages over batch size
    test_loss /= len(test_data)
    perplexity_loss /= len(test_data)

    # in NLP, we care about the perplexity of the model
    acc = round(correct / test_len, 4) 
    acc_5 = round(top_5 / test_len, 4)
    test_loss = round(test_loss, 4)

    if args.task == 'nlp':
        correct = math.exp(perplexity_loss)
        acc = correct

    logging.info('Rank {}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
          .format(rank, test_loss, correct, len(test_data.dataset), acc, acc_5))

    return test_loss, acc, acc_5, [correct, top_5, test_loss*float(test_len), test_len]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        #batch_size = target.size(0)

        #logging.info("====To get accuracy, top-k is {}, while shape is {}".format(maxk, output.shape))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res

class RandomParams(object):

    def __init__(self, ratio: float):
        self.ratio = ratio

    def get(self, params_indices: list):
        rng = random.Random()
        rng.seed(random.random() * 1234)
        indexes = [x for x in range(len(params_indices))]
        rng.shuffle(indexes)
        # print(indexes)

        part_len = int(math.floor(self.ratio * len(params_indices)))
        result = indexes[0: part_len]
        return [params_indices[i] for i in result]
