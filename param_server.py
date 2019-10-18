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
import torchvision.models as tormodels

parser = argparse.ArgumentParser()
# 集群基本信息的配置 - The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--ps_port', type=str, default='29500')
parser.add_argument('--this_rank', type=int, default=0)
parser.add_argument('--learners', type=str, default='1-2-3-4')

# 模型与数据集的配置 - The configuration of model and dataset
parser.add_argument('--data_dir', type=str, default='/tmp/')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--depth', type=int, default=18)
parser.add_argument('--data_set', type=str, default='cifar10')

# 训练时各种超参数的配置 - The configuration of different hyper-parameters for training
parser.add_argument('--timeout', type=float, default=10000.0)
parser.add_argument('--len_train_data', type=int, default=50000)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--test_bsz', type=int, default=256)
parser.add_argument('--stale_threshold', type=int, default=0)
parser.add_argument('--backend', type=str, default="gloo")
parser.add_argument('--display_step', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--force_read', type=bool, default=False)
parser.add_argument('--sleep_up', type=int, default=0)
parser.add_argument('--local', type=bool, default=False)
parser.add_argument('--local_split', type=int, default=1)
parser.add_argument('--sequential', type=int, default=0)
parser.add_argument('--filter_class', type=int, default=0)

args = parser.parse_args()
display_step = args.display_step

#torch.set_num_threads(1)
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
    if not args.local:
        for _ in workers:
            param_q.put(_tmp)
    else:
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
    learner_local_step = {l: 0 for l in workers}
    learner_cache_step = {l: 0 for l in workers}
    pendingWorkers = {}

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

            #print("Receive update from worker {}".format(rank_src))

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
            learner_local_step[rank_src] += 1

            handlerStart = time.time()
            # apply the update into the global model
            for idx, param in enumerate(model.parameters()):

                param.data -= (torch.from_numpy(delta_ws[idx]).cuda())

            handlerDur = time.time() - handlerStart

            #print ("====Handler duration is {}".format(handlerDur))
            global_update += 1
            currentMinStep = 9999999999

            # get the current minimum local steps
            for rankStep in learner_local_step.keys():
                currentMinStep = min(currentMinStep, learner_local_step[rankStep])

            stale = int(learner_local_step[rank_src] - currentMinStep)
            staleness_sum_epoch += stale
            staleness_sum_suqare_epoch += stale**2
            staleness += 1
            learner_staleness[rank_src] = staleness
            stale_stack.append(rank_src)

            # if the worker is within the staleness, then continue w/ local cache and do nothing
            # Otherwise, block it 
            if learner_local_step[rank_src] > args.stale_threshold + currentMinStep:
                pendingWorkers[rank_src] = learner_local_step[rank_src]
                # lock the worker
                f_staleness.write("Lock worker " + str(rank_src) + " with localStep " + str(pendingWorkers[rank_src]) +
                                        " , while globalStep is " + str(currentMinStep) + "\n")
            
            # if the local cache is too stale, then update it
            elif learner_cache_step[rank_src] < learner_local_step[rank_src] - args.stale_threshold or args.force_read:
                for idx, param in enumerate(model.parameters()):
                        dist.send(tensor=param.data.cpu(), dst=rank_src)

                # send out current minimum steps
                dist.send(tensor=torch.tensor([currentMinStep], dtype=torch.int).cpu(), dst=rank_src)
                learner_cache_step[rank_src] = currentMinStep
                #learner_local_step[rank_src]

            # release all pending requests, if the staleness does not exceed the staleness threshold in SSP 
            keys = list(pendingWorkers)
            handle_dur = time.time() - handle_start
            for pworker in keys:
                # check its staleness
                send_start = time.time()
                if pendingWorkers[pworker] <= args.stale_threshold + currentMinStep:
                    # start to send param, to avoid synchronization problem, first create a copy here?
                    for idx, param in enumerate(model.parameters()):
                        dist.send(tensor=param.data.cpu(), dst=pworker)

                    # send out current minimum step
                    dist.send(tensor=torch.tensor([currentMinStep], dtype=torch.int).cpu(), dst=pworker)
                    learner_cache_step[pworker] = currentMinStep    #learner_local_step[pworker]

                    if global_update % display_step == 0:
                        print("Handle Wight {} | Send {}".format(handle_dur, time.time() - send_start))
                    # remove from the pending workers
                    del pendingWorkers[pworker]

            # isStale = False
            # for stale_each_worker in learner_staleness:
            #     if (stale_each_worker not in stale_stack) & \
            #         (staleness - learner_staleness[stale_each_worker] > args.stale_threshold):
            #         isStale = True
            #         break

            # handle_dur = time.time() - handle_start
            
            # if not isStale:
            #     for i in range(len(stale_stack)):
            #         rank_wait = stale_stack.pop()
            #         # 相应learner下次更新的staleness - SSP: staleness upadate
            #         learner_staleness[rank_wait] = staleness

            #         send_start = time.time()
            #         for idx, param in enumerate(model.parameters()):
            #             dist.send(tensor=param.data.cpu(), dst=rank_wait)
            #         if global_update % display_step == 0:
            #             print("Handle Wight {} | Send {}".format(handle_dur, time.time() - send_start))
            # else:
            #     continue

            # #print('Done From Rank {}, Staleness {}!'
            # #      .format(rank_src, stale))
            # # epoch, rank, batch size, stale
            # f_staleness.write(str(epoch_count) +
            #             "\t" + str(rank_src) +
            #             "\t" + str(batch_size) +
            #             "\t" + str(stale) + '\n')

            # once reach an epoch, count the average train loss
            if(data_size_epoch >= args.len_train_data):
                e_epoch_time = time.time()
                #variance of stale
                diversity_stale = (staleness_sum_suqare_epoch/iteration_in_epoch)\
                                 - (staleness_sum_epoch/iteration_in_epoch)**2
                staleness_sum_suqare_epoch = 0
                staleness_sum_epoch = 0
                test_loss, test_acc = 0, 0
                #test_model(dist.get_rank(), model, test_data, criterion=criterion)
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
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
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
    elif args.data_set == 'imagenet':
        train_t, test_t = get_data_transform('imagenet')
        test_dataset = datasets.ImageNet(args.data_dir, split='train', download=True, transform=train_t)

        model = tormodels.__dict__[args.model]()
    else:
        print('DataSet must be {} or {}!'.format('Mnist', 'Cifar'))
        sys.exit(-1)

    if torch.cuda.is_available():
        model = model.cuda()

    test_data = DataLoader(test_dataset, batch_size=100, shuffle=True)

    print("====PS: finish loading test_data")

    if not args.local:
        world_size = len(str(args.learners).split('-')) + 1
    else:
        world_size = 2
        
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
