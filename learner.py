# -*- coding: utf-8 -*-
# Thanks to qihua Zhou

import argparse
import math
import os
import sys
import time
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import threading

import numpy as np
import torch
import torch.distributed as dist
from utils.divide_data import partition_dataset, select_dataset, DataPartitioner
from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import MySGD, test_model
from torch.autograd import Variable
from torchvision import datasets

parser = argparse.ArgumentParser()
# 集群基本信息的配置 - The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--ps_port', type=str, default='29500')
parser.add_argument('--this_rank', type=int, default=1)
parser.add_argument('--learners', type=str, default='1-2')

# 模型与数据集的配置 - The configuration of model and dataset
parser.add_argument('--data_dir', type=str, default='../../data')
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--model', type=str, default='MnistCNN')
parser.add_argument('--depth', type=int, default=56)
parser.add_argument('--data_set', type=str, default='cifar10')

# 训练时各种超参数的配置 - The configuration of different hyper-parameters for training
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_bsz', type=int, default=100)
parser.add_argument('--heterogeneity', type=float, default=1.0)
parser.add_argument('--hetero_allocation', type=str, default='1.0-1.0-1.0-1.0-1.0-1.0')
parser.add_argument('--backend', type=str, default="gloo")
parser.add_argument('--display_step', type=int, default=100)
parser.add_argument('--upload_epho', type=int, default=1)

args = parser.parse_args()
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def unbalanced_partition_dataset(dataset, hetero):
    """ Partitioning Data """
    computing_capacity = [float(v) for v in hetero.split('-')]
    norm_factor = sum(computing_capacity)
    partition_sizes = [v/norm_factor for v in computing_capacity]
    partition = DataPartitioner(dataset, partition_sizes)
    return partition

def run(rank, model, train_data, test_data, queue, param_q, stop_flag):
    print("====Worker: Start running")
    # 获取ps端传来的模型初始参数 - Fetch the initial parameters from the server side (we called it parameter_server)
    while True:
        if not param_q.empty():
            param_dict = param_q.get()
            tmp = OrderedDict(map(lambda item: (item[0], torch.from_numpy(item[1])),
                                  param_dict.items()))
            model.load_state_dict(tmp)
            break
    print('Model recved successfully!')

    if args.model == 'MnistCNN':
        optimizer = MySGD(model.parameters(), lr=0.001, momentum=0.5)
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        optimizer = MySGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss().cuda()

    print('Begin!')
    local_step = 0
    startTime = time.time()

    for epoch in range(int(args.epochs)):
        model.train()
        epoch_train_loss = 0
        epoch_train_batch = 0

        for batch_idx, (data, target) in enumerate(train_data):
            it_start = time.time()
            data, target = Variable(data).cuda(), Variable(target).cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            delta_ws = optimizer.get_delta_w()

            it_end = time.time()
            it_duration = it_end - it_start
            # mimic the heterogeneity through delay
            # The "heterogeneity" is a ratio of the maximum computing capacity
            sleep_time = (1.0 / args.heterogeneity - 1.0) * it_duration
            time.sleep(sleep_time)

            local_step += 1
            epoch_train_loss += loss.data.item()
            epoch_train_batch += 1

            # noinspection PyBroadException
            try:  # 捕获异常，异常来源于ps进程的停止 - Capture the exception caused by the shutdown of parameter_server
                send_dur = 0
                rece_dur = 0
                if local_step % args.upload_epho == 0:
                    send_start = time.time()
                    if delta_ws:
                        queue.put({
                            rank: [[v.cpu().numpy() for v in delta_ws], loss.data.cpu().numpy(), np.array(args.batch_size), False]
                        })

                    send_dur = time.time() - send_start

                    rece_start = time.time()
                    for idx, param in enumerate(model.parameters()):
                        tmp_tensor = torch.zeros_like(param.data).cpu()
                        dist.recv(tensor=tmp_tensor, src=0)
                        param.data = tmp_tensor.cuda()
                    rece_dur = time.time() - rece_start

                else:
                    for idx, param in enumerate(model.parameters()):
                        param.data -= delta_ws[idx]

                if local_step % args.display_step == 0:
                        print('LocalStep {}, CumulTime {}, Epoch {}, Batch {}/{}, Loss:{} | TotalTime {} | Comptime: {} | SendTime: {} | ReceTime: {}'
                            .format(local_step, time.time() - startTime, epoch, batch_idx, len(train_data), round(loss.data.item(), 4), round(time.time() - it_start, 2), round(it_duration, 2), round(send_dur,2), round(rece_dur, 2)))

            except Exception as e:
                print(str(e))
                print('Should Stop: {}!'.format(stop_flag.value))
                break

        # 训练结束后进行top 1的test - Check the top 1 test accuracy after training
        #print("test Model:", epoch)
        print("For epoch {}, training loss {}".format(epoch, epoch_train_loss/float(epoch_train_batch)))
        test_model(rank, model, test_data, criterion=criterion)
        #if stop_flag.value:
        #    break
    queue.put({rank: [[], [], [], True]})
    print("Worker {} has completed epoch {}!".format(args.this_rank, epoch))

def init_myprocesses(rank, size, model,
                   train_dataset, test_dataset,
                   q, param_q, stop_flag,
                   fn, backend):
    print("====Worker: init_myprocesses")
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, model, train_dataset, test_dataset, q, param_q, stop_flag)

def capture_stop(stop_signal, flag: Value):
    while True:
        if not stop_signal.empty():
            flag.value = True
            print('Time Up! Stop: {}!'.format(flag.value))
            break


if __name__ == "__main__":

    """
    判断使用的模型，MnistCNN或者是AlexNet - Check the modle type, here we support the flat MnistCNN and Alexnet supported by PyTorch
    模型不同，数据集、数据集处理方式、优化函数、损失函数、参数等都不一样 - Different operations need to be done according to model and dataset
    """
    if args.data_set == 'Mnist':
        model = MnistCNN()

        train_transform, test_transform = get_data_transform('mnist')

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                       transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                      transform=test_transform)

        train_bsz = args.batch_size
        test_bsz = args.test_bsz

    elif args.data_set == 'cifar10':
        train_transform, test_transform = get_data_transform('cifar')
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                        transform=test_transform)

        train_bsz = args.batch_size
        test_bsz = args.test_bsz

        if args.model == "alexnet":
            model = AlexNetForCIFAR()
        elif args.model == "vgg":
            model = VGG(args.depth)
        elif args.model == "resnet":
            model = ResNet(args.depth)
        elif args.model == "googlenet":
            model = GoogLeNet()
        else:
            print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
            sys.exit(-1)
    else:
        print('DataSet must be {} or {}!'.format('Mnist', 'Cifar'))
        sys.exit(-1)

    if torch.cuda.is_available():
        model = model.cuda()

    workers = [int(v) for v in str(args.learners).split('-')]
    train_data = partition_dataset(train_dataset, workers)
    #train_data = unbalanced_partition_dataset(train_dataset, args.hetero_allocation)
    test_data = partition_dataset(test_dataset, workers)

    this_rank = args.this_rank
    train_data = select_dataset(workers, this_rank, train_data, batch_size=train_bsz)
    test_data = select_dataset(workers, this_rank, test_data, batch_size=test_bsz)

    world_size = len(workers) + 1


    class MyManager(BaseManager):
        pass

    MyManager.register('get_queue')
    MyManager.register('get_param')
    MyManager.register('get_stop_signal')
    manager = MyManager(address=(args.ps_ip, 5000), authkey=b'queue')
    manager.connect()

    q = manager.get_queue()  # 更新参数使用的队列 - queue for parameter_server signal process
    param_q = manager.get_param()  # 接收初始模型参数使用的队列 - init
    stop_signal = manager.get_stop_signal()  # 接收停止信号使用的队列 - stop

    stop_flag = Value(c_bool, False)
    init_myprocesses(this_rank, world_size,
                                          model,
                                          train_data, test_data,
                                          q, param_q, stop_flag,
                                          run, args.backend)

    # 开启一个进程捕获ps的stop信息 - parameter_server signal process handler
    # stop_p = Process(target=capture_stop,
    #                  args=(stop_signal, stop_flag))

    # p = Process(target=init_myprocesses, args=(this_rank, world_size,
    #                                               model,
    #                                               train_data, test_data,
    #                                               q, param_q, stop_flag,
    #                                               run, 'gloo'))
    # p.start()
    # stop_p.start()
    # print("====Client: Start to join processes")
    # p.join()
    # stop_p.join()
