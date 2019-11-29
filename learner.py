# -*- coding: utf-8 -*-
# Thanks to qihua Zhou

import argparse
import math
import os, shutil
import sys
import time
import datetime
import logging
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import threading
import random

import numpy as np
import torch
import torch.distributed as dist
from utils.divide_data import partition_dataset, select_dataset, DataPartitioner
from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import MySGD, test_model
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as tormodels

parser = argparse.ArgumentParser()
# The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--ps_port', type=str, default='29500')
parser.add_argument('--this_rank', type=int, default=1)
parser.add_argument('--learners', type=str, default='1-2-3-4')
parser.add_argument('--total_worker', type=int, default=0)

# The configuration of model and dataset
parser.add_argument('--data_dir', type=str, default='/tmp/')
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--client_path', type=str, default='/tmp/client.cfg')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--depth', type=int, default=18)
parser.add_argument('--data_set', type=str, default='cifar10')
parser.add_argument('--sample_mode', type=str, default='random')

# The configuration of different hyper-parameters for training
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_bsz', type=int, default=256)
parser.add_argument('--heterogeneity', type=float, default=1.0)
parser.add_argument('--hetero_allocation', type=str, default='1.0-1.0-1.0-1.0-1.0-1.0')
parser.add_argument('--backend', type=str, default="gloo")
parser.add_argument('--display_step', type=int, default=20)
parser.add_argument('--upload_epoch', type=int, default=1)
parser.add_argument('--stale_threshold', type=int, default=0)
parser.add_argument('--sleep_up', type=int, default=0)
parser.add_argument('--force_read', type=bool, default=False)
parser.add_argument('--test_interval', type=int, default=999999)
parser.add_argument('--resampling_interval', type=int, default=99999999)
parser.add_argument('--sequential', type=int, default=0)
parser.add_argument('--single_sim', type=int, default=0)
parser.add_argument('--filter_class', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--model_avg', type=bool, default=False)
parser.add_argument('--input_dim', type=int, default=0)
parser.add_argument('--output_dim', type=int, default=0)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--dump_epoch', type=int, default=100)
parser.add_argument('--decay_factor', type=float, default=0.9)
parser.add_argument('--decay_epoch', type=float, default=500)
parser.add_argument('--threads', type=str, default=str(torch.get_num_threads()))
parser.add_argument('--eval_interval', type=int, default=5)

args = parser.parse_args()
#torch.cuda.set_device(0)


dirPath = '/tmp/torch/'
if not os.path.isdir(dirPath):
    os.mkdir(dirPath)

logFile = dirPath + 'log_'+str(args.this_rank) + '_'+ str(datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S'))

def init_logging():
    files = [logFile, '/tmp/sampleDistribution']
    for file in files:
        with open(file, "w") as fout:
            pass

init_logging()

logging.basicConfig(filename=logFile,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

entire_train_data = None
os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
os.environ['OMP_NUM_THREADS'] = args.threads
os.environ['MKL_NUM_THREADS'] = args.threads

torch.set_num_threads(int(args.threads))

clientIdToRun = args.this_rank

logging.info("===== Experiment start =====")

def unbalanced_partition_dataset(dataset, hetero):
    """ Partitioning Data """
    computing_capacity = [float(v) for v in hetero.split('-')]
    norm_factor = sum(computing_capacity)
    partition_sizes = [v/norm_factor for v in computing_capacity]
    partition = DataPartitioner(dataset, partition_sizes)
    return partition

def run(rank, model, train_data, test_data, queue, param_q, stop_flag, client_cfg):
    print("====Worker: Start running")
    # Fetch the initial parameters from the server side (we called it parameter_server)
    while True:
        if not param_q.empty():
            param_dict = param_q.get()
            tmp = OrderedDict(map(lambda item: (item[0], torch.from_numpy(item[1])),
                                  param_dict.items()))
            model.load_state_dict(tmp)
            break

    print('Model recved successfully!')

    LinearModel = ['Logistic', 'Linear', 'SVM']

    if args.model == 'MnistCNN':
        optimizer = MySGD(model.parameters(), lr=args.learning_rate, momentum=0.5)
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        optimizer = MySGD(model.parameters(), lr=args.learning_rate, momentum=0, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss().cuda()

    print('Begin!')
    logging.info('\n' + repr(args) + '\n')

    local_step = 0
    startTime = time.time()
    globalMinStep = 0
    last_test = 0
    paramDic = {}

    logDir = "/tmp/" + args.model
    if os.path.isdir(logDir):
        shutil.rmtree(logDir)

    os.mkdir(logDir)

    for idx, (name, param) in enumerate(model.named_parameters()):
        paramDic[idx] = repr(name) + '\t' + repr(param.size())
        
        with open(logDir + "/" + str(idx), 'w') as f:
            f.write(repr(idx) + ': ' + repr(name) + '\t'+ repr(param.size()) +'\n')

    random.seed(rank)

    train_data_itr = [iter(data) for data in train_data]
    learning_rate = args.learning_rate
    last_push_time = time.time()
    currentClientId = clientIdToRun
    reloadClientData = False

    sleepForCompute = 0 if currentClientId not in client_cfg else client_cfg[currentClientId][0]
    sleepForCommunicate = 0 if currentClientId not in client_cfg else client_cfg[currentClientId][1]

    for epoch in range(int(args.epochs)):
        model.train()
        epoch_train_loss = 0
        epoch_train_batch = 0
        local_trained = 0

        if epoch % args.decay_epoch == 0:
            learning_rate = max(0.001, learning_rate * args.decay_factor)

        if reloadClientData:
            train_data = []
            train_data.append(select_dataset(workers, nextClientId, entire_train_data, batch_size=train_bsz))
            train_data_itr = [iter(data) for data in train_data]
            reloadClientData = False

        for batch_idx in range(len(train_data[-1])):
            # for every mini-batch/iteration
            local_trained += args.batch_size

            it_start = time.time()
            delta_wss = []

            for machineId in range(len(train_data)):
                # simulate multiple workers
                try:
                    (data, target) = next(train_data_itr[machineId])
                except StopIteration:
                    # start from the beginning again
                    train_data_itr[machineId] = iter(train_data[machineId])
                    (data, target) = next(train_data_itr[machineId])

                # LR is not compatible with the image input as of now, thus needs to reformat it 
                if args.data_set == 'emnist' and args.model in LinearModel:
                    data = data.view(-1, 28 * 28)

                data, target = Variable(data).cuda(), Variable(target).cuda()
                optimizer.zero_grad()

                forwardS = time.time()
                output = model(data)
                forD = time.time() - forwardS

                loss = criterion(output, target)
                loss.backward()

                startGet = time.time()
                delta_ws = optimizer.get_delta_w(learning_rate)
                delta_wss.append(delta_ws)

                deltaDur = time.time() - startGet

            it_end = time.time()
            it_duration = it_end - it_start

            if sleepForCompute != 0:
                #sleep_time = random.uniform(0, args.sleep_up)/1000.0
                time.sleep(sleepForCompute * it_duration)

            local_step += 1
            epoch_train_loss += loss.data.item()
            epoch_train_batch += 1

            try:  # Capture the exception caused by the shutdown of parameter_server
                send_dur = 0
                rece_dur = 0

                for idx, param in enumerate(model.parameters()):
                    for delta_w in delta_wss:
                        param.data -= delta_w[idx].cuda()

                if local_step % args.upload_epoch == 0:
                    send_start = time.time()

                    # push update to the PS
                    if delta_ws:
                        training_speed = local_trained/(time.time() - last_push_time)

                        if sleepForCommunicate != 0:
                            time.sleep(sleepForCommunicate/2.0)

                        if not args.model_avg:
                            queue.put_nowait({
                                rank: [[v.cpu().numpy() for v in delta_ws], loss.data.cpu().numpy(), 
                                    np.array(local_trained), False, currentClientId, training_speed]
                            })
                        else:
                            queue.put_nowait({
                                rank: [[param.data.cpu().numpy() for param in model.parameters()], 
                                    loss.data.cpu().numpy(), np.array(local_trained), False, currentClientId, training_speed]
                            })

                        local_trained = 0
                        last_push_time = time.time()
                        logging.info("====Push updates")

                    send_dur = time.time() - send_start

                    rece_start = time.time()

                    if globalMinStep < local_step - args.stale_threshold or args.force_read:

                        if epoch % args.eval_interval == 0:
                            test_loss, acc = test_model(rank, model, test_data, criterion=criterion)
                            logging.info("Before aggregation with client {}, for epoch {}, CumulTime {}, training loss {}, test_loss {}, test_accuracy {} \n".format(currentClientId, epoch, time.time() - startTime, epoch_train_loss/float(epoch_train_batch), test_loss, acc))

                        rece_start = time.time()
                        # local cache is too stale
                        for idx, param in enumerate(model.parameters()):
                            tmp_tensor = torch.zeros_like(param.data).cpu()
                            dist.recv(tensor=tmp_tensor, src=0)
                            param.data = tmp_tensor.cuda()

                        # receive current minimum step, and the clientId for next training
                        step_tensor = torch.zeros([2], dtype=torch.int).cpu()
                        dist.recv(tensor=step_tensor, src=0)
                        globalMinStep, nextClientId = step_tensor[0].item(), step_tensor[1].item()

                        if sleepForCommunicate != 0:
                            time.sleep(sleepForCommunicate/2.0)

                        # reload the training data for the specific clientId
                        if nextClientId != currentClientId:
                            logging.info('====Training switch from clientId {} to clientId {}'.format(currentClientId, nextClientId))
                            currentClientId = nextClientId
                            reloadClientData = True
                            sleepForCompute = 0 if currentClientId not in client_cfg else client_cfg[currentClientId][0]
                            sleepForCommunicate = 0 if currentClientId not in client_cfg else client_cfg[currentClientId][1]

                    rece_dur = time.time() - rece_start

                    if epoch % args.eval_interval == 0:
                        test_loss, acc = test_model(rank, model, test_data, criterion=criterion)
                        logging.info("After aggregation, for epoch {}, CumulTime {}, training loss {}, test_loss {}, test_accuracy {} \n".format(epoch, time.time() - startTime, epoch_train_loss/float(epoch_train_batch), test_loss, acc))
                        model.train()

                logging.info('LocalStep {}, CumulTime {}, Epoch {}, Batch {}/{}, Loss:{} | TotalTime {} | Comptime: {} | SendTime: {} | ReceTime: {} | Sleep: {} | staleness: {} \n'
                            .format(local_step, time.time() - startTime, epoch, batch_idx, len(train_data[0]), round(loss.data.item(), 4), round(time.time() - it_start, 4), round(it_duration, 4), round(send_dur,4), round(rece_dur, 4), sleepForCompute, local_step - globalMinStep))

            except Exception as e:
                print(str(e))
                logging.info("====Error: " + str(e) + '\n')
                print('Should Stop: {}!'.format(stop_flag.value))
                break

            if time.time() - last_test > args.test_interval:
                last_test = time.time()
                test_loss, acc = test_model(rank, model, test_data, criterion=criterion)
                logging.info("For epoch {}, CumulTime {}, training loss {}, test_loss {}, test_accuracy {} \n".format(epoch, time.time() - startTime, epoch_train_loss/float(epoch_train_batch), test_loss, acc))
                model.train()

        # Check the top 1 test accuracy after training
        print("For epoch {}, training loss {}, CumulTime {}, local Step {} ".format(epoch, epoch_train_loss/float(epoch_train_batch), time.time() - startTime, local_step))
        #test_loss, acc = test_model(rank, model, test_data, criterion=criterion)
        #logging.info("For epoch {}, CumulTime {}, training loss {}, test_loss {}, test_accuracy {} \n".format(epoch, time.time() - startTime, epoch_train_loss/float(epoch_train_batch), test_loss, acc))
        last_test = time.time()

        #if stop_flag.value:
        #    break
    queue.put({rank: [[], [], [], True, -1, -1]})
    print("Worker {} has completed epoch {}!".format(args.this_rank, epoch))

def report_data_info(rank, queue, entire_train_data):
    # report data information to the clientSampler master
    queue.put({
        rank: [entire_train_data.getDistance(), entire_train_data.getSize()]
    })

    # get the partitionId to run
    clientIdToRun = torch.zeros([1], dtype=torch.int).cpu()
    dist.recv(tensor=clientIdToRun, src=0)
    return clientIdToRun.item()

def init_myprocesses(rank, size, model,
                   train_dataset, test_dataset,
                   q, param_q, stop_flag,
                   fn, backend, client_cfg):
    print("====Worker: init_myprocesses")
    fn(rank, model, train_dataset, test_dataset, q, param_q, stop_flag, client_cfg)

def capture_stop(stop_signal, flag: Value):
    while True:
        if not stop_signal.empty():
            flag.value = True
            print('Time Up! Stop: {}!'.format(flag.value))
            break

class MyManager(BaseManager):
        pass

def init_dataset():

    if args.data_set == 'Mnist':
        model = MnistCNN()
        train_transform, test_transform = get_data_transform('mnist')

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                       transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                      transform=test_transform)
    elif args.data_set == 'cifar10':
        train_transform, test_transform = get_data_transform('cifar')
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                        transform=test_transform)
        if args.model == "alexnet":
            model = AlexNet()
        elif args.model == "vgg":
            model = VGG(args.depth)
        elif args.model == "resnet":
            model = ResNet(args.depth)
        elif args.model == "googlenet":
            model = GoogLeNet()
        elif args.model == "lenet":
            model = LeNet()
        else:
            print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
            sys.exit(-1)

    elif args.data_set == "imagenet":
        train_transform, test_transform = get_data_transform('imagenet')
        train_dataset = datasets.ImageNet(args.data_dir, split='train', download=True, transform=train_transform)
        test_dataset = datasets.ImageNet(args.data_dir, split='train', download=True, transform=train_transform)

        model = tormodels.__dict__[args.model]()
    elif args.data_set == 'emnist':
        test_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=False, download=True, transform=transforms.ToTensor())
        train_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=True, download=True, transform=transforms.ToTensor())

        if args.model == "Logistic":
            model = LogisticRegression(args.input_dim, args.output_dim)
        elif args.model == "alexnet":
            model = AlexNetForMnist(47)
        elif args.model == "vgg":
            model = VGG(args.depth, args.output_dim, 1)
        elif args.model == "resnet":
            model = ResNet(args.depth, args.output_dim, 1)
        elif args.model == "lenet":
            model = LeNetForMNIST(args.output_dim)
        else:
            print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
            sys.exit(-1)
    else:
        print('DataSet must be {} or {}!'.format('Mnist', 'Cifar'))
        sys.exit(-1)

    if torch.cuda.is_available():
        model = model.cuda()

    # initiate the device information - normalized computation speed by enforcing sleeping, bandwidth
    client_cfg = {}

    if os.path.exists(args.client_path):
        with open(args.client_path, 'r') as fin:
            for line in fin.readlines():
                items = line.strip().split()
                clientId, compute, commu = int(items[0]), float(items[1]), float(items[2])
                client_cfg[clientId] = [compute, commu]

    return model, train_dataset, test_dataset, client_cfg

if __name__ == "__main__":

    torch.manual_seed(args.this_rank)

    train_bsz = args.batch_size
    test_bsz = args.test_bsz

    model, train_dataset, test_dataset, client_cfg = init_dataset()

    splitTrainRatio = []
    workers = [int(v) for v in str(args.learners).split('-')]

    # Initialize PS - client communication channel
    world_size = len(workers) + 1
    this_rank = args.this_rank

    MyManager.register('get_queue')
    MyManager.register('get_param')
    MyManager.register('get_stop_signal')
    manager = MyManager(address=(args.ps_ip, 5000), authkey=b'queue')
    manager.connect()

    q = manager.get_queue()  # queue for parameter_server signal process
    param_q = manager.get_param()  # init
    stop_signal = manager.get_stop_signal()  # stop
    dist.init_process_group(args.backend, rank=this_rank, world_size=world_size)

    # Split the dataset
    # total_worker != 0 indicates we create more virtual clients for simulation
    if args.total_worker > 0:
        workers = [i for i in range(1, args.total_worker + 1)]

    entire_train_data = partition_dataset(train_dataset, workers, splitTrainRatio, args.sequential, filter_class=args.filter_class)
    clientIdToRun = report_data_info(this_rank, q, entire_train_data)

    train_datas = []

    if args.single_sim != 0:
        for rank in workers:
            train_datas.append(select_dataset(workers, rank, entire_train_data, batch_size=train_bsz))
    else:
        train_datas.append(select_dataset(workers, clientIdToRun, entire_train_data, batch_size=train_bsz))

    testWorkers = workers
    splitTestRatio = []

    test_data = partition_dataset(test_dataset, [1], splitTestRatio)
    test_data = select_dataset(testWorkers, this_rank, test_data, batch_size=test_bsz, istest=True)

    stop_flag = Value(c_bool, False)
    init_myprocesses(this_rank, world_size, model, train_datas, test_data,
                                          q, param_q, stop_flag,
                                          run, args.backend, client_cfg)
