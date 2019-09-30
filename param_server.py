# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
import time
from collections import OrderedDict
from multiprocessing.managers import BaseManager
import torch
import torch.distributed as dist
from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import test_model
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader
from torchvision import datasets

parser = argparse.ArgumentParser()
# 集群基本信息的配置 - The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--ps_port', type=str, default='29500')
parser.add_argument('--this_rank', type=int, default=0)
parser.add_argument('--learners', type=str, default='1-2')

# 模型与数据集的配置 - The configuration of model and dataset
parser.add_argument('--data_dir', type=str, default='../../data')
parser.add_argument('--model', type=str, default='MnistCNN')
parser.add_argument('--depth', type=int, default=56)
parser.add_argument('--data_set', type=str, default='cifar10')

# 训练时各种超参数的配置 - The configuration of different hyper-parameters for training
parser.add_argument('--timeout', type=float, default=10000.0)
parser.add_argument('--len_train_data', type=int, default=60000)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--stale_threshold', type=int, default=1000)
parser.add_argument('--backend', type=str, default="gloo")
parser.add_argument('--display_step', type=int, default=100)

args = parser.parse_args()
display_step = args.display_step

def run(model, test_data, queue, param_q, stop_signal):
    if args.model == 'MnistCNN':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    print("====PS: get in run()")
    # 参数中的tensor转成numpy - convert gradient tensor to numpy structure
    tmp = map(lambda item: (item[0], item[1].numpy), model.state_dict().items())
    _tmp = OrderedDict(map(lambda item: (item[0], item[1].cpu().numpy()), model.state_dict().items()))
    workers = [int(v) for v in str(args.learners).split('-')]
    for _ in workers:
        param_q.put(_tmp)
    print('Model Sent Finished!')

    print('Begin!')

    epoch_train_loss = 0
    iteration_in_epoch = 0
    data_size_epoch = 0   # len(train_data), one epoch
    epoch_count = 0
    staleness_sum_suqare_epoch = 0
    staleness_sum_epoch = 0

    staleness = 0
    learner_staleness = {l: 0 for l in workers}
    s_time = time.time()
    epoch_time = s_time

    # In SSP, the fast workers have to wait the slowest worker a given duration
    # The fast worker exceeding the duration will be pushed into the queue to wait
    stale_stack = []

    trainloss_file = '/tmp/trainloss' + args.model + '.txt'
    staleness_file = '/tmp/staleness' + args.model + ".txt"

    f_trainloss = open(trainloss_file, 'w')
    f_staleness = open(staleness_file, 'w')
    global_update = 0

    while True:
        if not queue.empty():
            handle_start = time.time()
            tmp_dict = queue.get()
            rank_src = list(tmp_dict.keys())[0]
            isWorkerEnd = tmp_dict[rank_src][3]

            if isWorkerEnd:
                print("Worker {} has completed all its data computation!".format(rank_src))
                learner_staleness.pop(rank_src)
                if (len(learner_staleness) == 0):
                    f_trainloss.close()
                    f_staleness.close()
                    stop_signal.put(1)
                    print('Epoch is done: {}'.format(epoch_count))
                    break
                continue

            delta_ws = tmp_dict[rank_src][0]  # 取出字典：k：参数索引 v：delta_w - dictionary: key parameter index, value: v：delta_w(gradient)
            iteration_loss = tmp_dict[rank_src][1]
            batch_size = tmp_dict[rank_src][2]

            iteration_in_epoch += 1
            epoch_train_loss += iteration_loss
            data_size_epoch += batch_size

            for idx, param in enumerate(model.parameters()):
                param.data -= torch.from_numpy(delta_ws[idx]).cuda()
            global_update += 1

            stale = int(staleness - learner_staleness[rank_src])
            staleness_sum_epoch += stale
            staleness_sum_suqare_epoch += stale**2
            staleness += 1
            learner_staleness[rank_src] = staleness
            stale_stack.append(rank_src)

            # judge if the staleness exceed the staleness threshold in SSP
            isStale = False
            for stale_each_worker in learner_staleness:
                if (stale_each_worker not in stale_stack) & \
                    (staleness - learner_staleness[stale_each_worker] > args.stale_threshold):
                    isStale = True
                    break

            handle_dur = time.time() - handle_start
            
            if not isStale:
                for i in range(len(stale_stack)):
                    rank_wait = stale_stack.pop()
                    # 相应learner下次更新的staleness - SSP: staleness upadate
                    learner_staleness[rank_wait] = staleness

                    send_start = time.time()
                    for idx, param in enumerate(model.parameters()):
                        dist.send(tensor=param.data.cpu(), dst=rank_wait)
                    if global_update % display_step == 0:
                        print("Handle Wight {} | Send {}".format(handle_dur, time.time() - send_start))
            else:
                continue

            #print('Done From Rank {}, Staleness {}!'
            #      .format(rank_src, stale))
            # epoch, rank, batch size, stale
            f_staleness.write(str(epoch_count) +
                        "\t" + str(rank_src) +
                        "\t" + str(batch_size) +
                        "\t" + str(stale) + '\n')

            # once reach an epoch, count the average train loss
            if(data_size_epoch >= args.len_train_data):
                e_epoch_time = time.time()
                #variance of stale
                diversity_stale = (staleness_sum_suqare_epoch/iteration_in_epoch)\
                                 - (staleness_sum_epoch/iteration_in_epoch)**2
                staleness_sum_suqare_epoch = 0
                staleness_sum_epoch = 0
                test_loss, test_acc = test_model(dist.get_rank(), model, test_data, criterion=criterion)
                # rank, trainloss, variance of stalness, time in one epoch, time till now
                f_trainloss.write(str(args.this_rank) +
                                  "\t" + str(epoch_train_loss/float(iteration_in_epoch)) +
                                  "\t" + str(diversity_stale) +
                                  "\t" + str(e_epoch_time - epoch_time) +
                                  "\t" + str(e_epoch_time - s_time) +
                                  "\t" + str(epoch_count) +
                                  "\t" + str(test_acc) + '\n')
                f_trainloss.flush()
                f_staleness.flush()
                iteration_in_epoch = 0
                epoch_count += 1
                epoch_train_loss = 0
                data_size_epoch = 0
                epoch_time = e_epoch_time

            # The training stop
            if(epoch_count >= args.epochs):
                f_trainloss.close()
                f_staleness.close()
                stop_signal.put(1)
                print('Epoch is done: {}'.format(epoch_count))
                break

        e_time = time.time()
        if (e_time - s_time) >= float(args.timeout):
            f_trainloss.close()
            f_staleness.close()
            stop_signal.put(1)
            print('Time up: {}, Stop Now!'.format(e_time - s_time))
            break


def init_myprocesses(rank, size, model, test_data, queue, param_q, stop_signal, fn, backend):
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(model, test_data, queue, param_q, stop_signal)

if __name__ == "__main__":

    # 随机数设置 - Random
    manual_seed = args.this_rank
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    if args.data_set == 'Mnist':
        model = MnistCNN()
        train_t, test_t = get_data_transform('mnist')
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                      transform=test_t)
    elif args.data_set == 'cifar10':
        train_t, test_t = get_data_transform('cifar')
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                        transform=test_t)
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

    test_data = DataLoader(test_dataset, batch_size=100, shuffle=True)

    print("====PS: finish loading test_data")
    world_size = len(str(args.learners).split('-')) + 1
    this_rank = args.this_rank

    queue = Queue()
    param = Queue()
    stop_or_not = Queue()

    class MyManager(BaseManager):
        pass

    MyManager.register('get_queue', callable=lambda: queue)
    MyManager.register('get_param', callable=lambda: param)
    MyManager.register('get_stop_signal', callable=lambda: stop_or_not)
    manager = MyManager(address=(args.ps_ip, 5000), authkey=b'queue')
    manager.start()
    
    q = manager.get_queue()  # 更新参数使用的队列 - queue for parameter_server signal process
    param_q = manager.get_param()  # 开始时传模型参数使用的队列 - init
    stop_signal = manager.get_stop_signal()  # 传停止信号使用的队列 - stop

    init_myprocesses(this_rank, world_size, model,test_data,
                                                  q, param_q, stop_signal, run, args.backend)
    #p = Process(target=init_myprocesses, args=(this_rank, world_size, model,test_data,
    #                                               q, param_q, stop_signal, run, "gloo"))
    #p.start()
    #p.join()
    manager.shutdown()
