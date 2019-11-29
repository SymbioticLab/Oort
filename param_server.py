# -*- coding: utf-8 -*-

import argparse
import os, shutil
import random
import sys
import time
import logging
from clientSampler import ClientSampler
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
from torchvision import datasets, transforms
import torchvision.models as tormodels

parser = argparse.ArgumentParser()
# The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--ps_port', type=str, default='29500')
parser.add_argument('--this_rank', type=int, default=0)
parser.add_argument('--learners', type=str, default='1-2-3-4')
parser.add_argument('--total_worker', type=int, default=0)

# The configuration of model and dataset
parser.add_argument('--data_dir', type=str, default='/tmp/')
parser.add_argument('--client_path', type=str, default='/tmp/client.cfg')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--depth', type=int, default=18)
parser.add_argument('--data_set', type=str, default='cifar10')
parser.add_argument('--sample_mode', type=str, default='random')

# The configuration of different hyper-parameters for training
parser.add_argument('--timeout', type=float, default=100000.0)
parser.add_argument('--len_train_data', type=int, default=50000)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--test_bsz', type=int, default=256)
parser.add_argument('--stale_threshold', type=int, default=0)
parser.add_argument('--backend', type=str, default="gloo")
parser.add_argument('--display_step', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--upload_epoch', type=int, default=1)
parser.add_argument('--resampling_interval', type=int, default=99999999)
parser.add_argument('--force_read', type=bool, default=False)
parser.add_argument('--sleep_up', type=int, default=0)
parser.add_argument('--sequential', type=int, default=0)
parser.add_argument('--single_sim', type=int, default=0)
parser.add_argument('--filter_class', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--model_avg', type=bool, default=False)
parser.add_argument('--input_dim', type=int, default=0)
parser.add_argument('--output_dim', type=int, default=47)
parser.add_argument('--test_interval', type=int, default=999999)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--dump_epoch', type=int, default=100)
parser.add_argument('--decay_factor', type=float, default=0.9)
parser.add_argument('--decay_epoch', type=float, default=500)
parser.add_argument('--threads', type=str, default=str(torch.get_num_threads()))
parser.add_argument('--eval_interval', type=int, default=5)

args = parser.parse_args()

logFile = '/tmp/log'+str(args.this_rank)
logging.basicConfig(filename=logFile,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

entire_train_data = None
sample_size_dic = {}

trainloss_file = '/tmp/trainloss' + args.model + '.txt'
staleness_file = '/tmp/staleness' + args.model + ".txt"
os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
os.environ['OMP_NUM_THREADS'] = args.threads
os.environ['MKL_NUM_THREADS'] = args.threads

torch.set_num_threads(int(args.threads))

def initiate_sampler_query(numOfClients):
    # Initiate the clientSampler 
    clientSampler = ClientSampler(args.sample_mode)
    collectedClients = 0
    initial_time = time.time()

    # In this simulation, we run data split on each worker, which amplifies the # of datasets
    # Waiting for the data information from clients, or timeout
    while collectedClients < numOfClients or (time.time() - initial_time) > 5000:
        if not queue.empty():
            tmp_dict = queue.get()
            rank_src = list(tmp_dict.keys())[0]
            distanceVec = tmp_dict[rank_src][0]
            sizeVec = tmp_dict[rank_src][1]

            for clientId, dis in enumerate(distanceVec):
                # since the worker rankId starts from 1, we also configure the initial dataId as 1
                clientSampler.registerClient(rank_src, clientId + 1, dis, sizeVec[clientId])

            collectedClients += 1

    return clientSampler

def run(model, test_data, queue, param_q, stop_signal, clientSampler):
    logging.info("====PS: get in run()")

    f_staleness = open(staleness_file, 'w')

    logDir = "/tmp/" + args.model

    if args.load_model:
        try:
            model.load_state_dict(torch.load(logDir+'/'+str(args.model)+'.pth.tar'))
            logging.info("====Load model successfully\n")
        except Exception as e:
            logging.info("====Error: Failed to load model due to {}\n".format(str(e)))
            pass
    
    if os.path.isdir(logDir):
        shutil.rmtree(logDir)
    os.mkdir(logDir)

    # convert gradient tensor to numpy structure
    tmp = map(lambda item: (item[0], item[1].numpy), model.state_dict().items())
    _tmp = OrderedDict(map(lambda item: (item[0], item[1].cpu().numpy()), model.state_dict().items()))
    workers = [int(v) for v in str(args.learners).split('-')]

    for _ in workers:
        param_q.put(_tmp)

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
    global_update = 0
    newEpoch = True

    logging.info(repr(clientSampler.getClientsInfo()))

    while True:
        if not queue.empty():
            try:
                handle_start = time.time()
                tmp_dict = queue.get()
                rank_src = list(tmp_dict.keys())[0]

                [iteration_loss, trained_size, isWorkerEnd, clientId, speed] = [tmp_dict[rank_src][i] for i in range(1, len(tmp_dict[rank_src]))]
                #clientSampler.registerSpeed(rank_src, clientId, speed)
                clientSampler.registerScore(clientId, 1.0 - clientSampler.getScore(rank_src, clientId))

                if isWorkerEnd:
                    print("Worker {} has completed all its data computation!".format(rank_src))
                    learner_staleness.pop(rank_src)
                    if (len(learner_staleness) == 0):
                        f_staleness.close()
                        stop_signal.put(1)
                        print('Epoch is done: {}'.format(epoch_count))
                        break
                    continue

                iteration_in_epoch += 1
                epoch_train_loss += iteration_loss
                data_size_epoch += trained_size
                learner_local_step[rank_src] += 1

                handlerStart = time.time()
                delta_ws = tmp_dict[rank_src][0]

                # fraction of total samples on this specific node 
                ratioSample = clientSampler.getSampleRatio(clientId, rank_src)

                # apply the update into the global model
                for idx, param in enumerate(model.parameters()):
                    if not args.model_avg:
                        param.data -= (torch.from_numpy(delta_ws[idx]).cuda())
                    else:
                        if newEpoch == 0:
                            param.data = (torch.from_numpy(delta_ws[idx]).cuda()) * ratioSample
                        else:
                            param.data += (torch.from_numpy(delta_ws[idx]).cuda()) * ratioSample

                newEpoch += 1

                handlerDur = time.time() - handlerStart
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


                # once reach an epoch, count the average train loss
                if global_update%len(workers) == 0:
                    e_epoch_time = time.time()
                    #variance of stale
                    diversity_stale = (staleness_sum_suqare_epoch/iteration_in_epoch)\
                                     - (staleness_sum_epoch/iteration_in_epoch)**2
                    staleness_sum_suqare_epoch = 0
                    staleness_sum_epoch = 0
                    test_loss, test_acc = 0, 0
                    epoch_count += args.upload_epoch

                    # rank, trainloss, variance of stalness, time in one epoch, time till now
                    logging.info(str(args.this_rank) +
                                      "\t" + str(epoch_train_loss/float(iteration_in_epoch)) +
                                      "\t" + str(diversity_stale) +
                                      "\t" + str(e_epoch_time - epoch_time) +
                                      "\t" + str(e_epoch_time - s_time) +
                                      "\t" + str(epoch_count) +
                                      "\t" + str(test_acc) + '\n')
                    f_staleness.flush()
                    iteration_in_epoch = 0
                    epoch_train_loss = 0
                    data_size_epoch = 0
                    epoch_time = e_epoch_time

                    # dump the model into file for backup
                    if epoch_count % args.dump_epoch == 0:
                        logging.info("====Try to dump the model")
                        torch.save(model.state_dict(), logDir+'/'+str(args.model)+'.pth.tar')

                    # resampling the clients if necessary
                    if epoch_count % args.resampling_interval == 0:
                        logging.info("====Try to resample clients with metrics {}".format(repr(clientSampler.getAllMetrics())))
                        sampledClients = clientSampler.resampleClients(len(workers), max(args.total_worker, len(workers)))
                        for i, w in enumerate(workers):
                            clientSampler.clientOnHost(sampledClients[i], w)

                # if the worker is within the staleness, then continue w/ local cache and do nothing
                # Otherwise, block it 
                if learner_local_step[rank_src] >= args.stale_threshold + currentMinStep:
                    pendingWorkers[rank_src] = learner_local_step[rank_src]
                    # lock the worker
                    logging.info("Lock worker " + str(rank_src) + " with localStep " + str(pendingWorkers[rank_src]) +
                                            " , while globalStep is " + str(currentMinStep) + "\n")
                
                # if the local cache is too stale, then update it
                elif learner_cache_step[rank_src] < learner_local_step[rank_src] - args.stale_threshold or args.force_read:
                    pendingWorkers[rank_src] = learner_local_step[rank_src]
                    
                # release all pending requests, if the staleness does not exceed the staleness threshold in SSP 
                handle_dur = time.time() - handle_start

                workersToSend = []

                for pworker in pendingWorkers.keys():
                    # check its staleness
                    if pendingWorkers[pworker] <= args.stale_threshold + currentMinStep:
                        # start to send param, to avoid synchronization problem, first create a copy here?
                        workersToSend.append(pworker)

                if len(workersToSend) > 0:
                    send_start = time.time()
                    for idx, param in enumerate(model.parameters()):
                        pendingSend = []    # working channels in communication
                        for worker in workersToSend:
                            pendingSend.append(dist.isend(tensor=(param.data).cpu(), dst=worker))
                            #dist.send(tensor=(param.data/float(normWeight)).cpu(), dst=worker)
                        for item in pendingSend:
                            item.wait()

                    pendingSend = []
                    for worker in workersToSend:
                        pendingSend.append(dist.isend(tensor=torch.tensor([currentMinStep, clientSampler.getCurrentClientId(worker)], dtype=torch.int).cpu(), dst=worker))
                        #dist.send(tensor=torch.tensor([currentMinStep, clientSampler.getCurrentClientId(worker)], dtype=torch.int).cpu(), dst=worker)
                        learner_cache_step[worker] = currentMinStep
                        # remove from the pending workers
                        del pendingWorkers[worker]

                    for item in pendingSend:
                        item.wait()

                    if global_update % args.display_step == 0:
                        logging.info("Handle Wight {} | Send {}".format(handle_dur, time.time() - send_start))

                    newEpoch = 0

                # The training stop
                if(epoch_count >= args.epochs * args.upload_epoch):
                    f_staleness.close()
                    stop_signal.put(1)
                    print('Epoch is done: {}'.format(epoch_count))
                    break

            except Exception as e:
                print("====Error: " + str(e) + '\n')
                logging.info("====Error: " + str(e) + '\n')

        e_time = time.time()
        if (e_time - s_time) >= float(args.timeout):
            f_staleness.close()
            stop_signal.put(1)
            print('Time up: {}, Stop Now!'.format(e_time - s_time))
            break

def init_myprocesses(rank, size, model, test_data, queue, param_q, stop_signal, fn, backend):
    dist.init_process_group(backend, rank=rank, world_size=size)

    # After collecting all data information, then decide the clientId to run
    workerRanks = [int(v) for v in str(args.learners).split('-')]
    clientSampler = initiate_sampler_query(len(workerRanks))
    
    for wrank in workerRanks:
        nextClientIdToRun = clientSampler.nextClientIdToRun(hostId=wrank)
        clientSampler.clientOnHost(nextClientIdToRun, wrank)
        dist.send(tensor=torch.tensor([nextClientIdToRun], dtype=torch.int).cpu(), dst=wrank)

    # Start the PS service
    fn(model, test_data, queue, param_q, stop_signal, clientSampler)

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

    return model, train_dataset, test_dataset

class MyManager(BaseManager):
        pass

if __name__ == "__main__":

    with open(logFile, 'w') as f:
        pass
    # Control the global random
    manual_seed = args.this_rank
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    model, train_dataset, test_dataset = init_dataset()

    test_data = DataLoader(test_dataset, batch_size=100, shuffle=True)

    print("====PS: finish loading test_data")
    world_size = len(str(args.learners).split('-')) + 1
        
    this_rank = args.this_rank

    queue = Queue()
    param = Queue()
    stop_or_not = Queue()

    MyManager.register('get_queue', callable=lambda: queue)
    MyManager.register('get_param', callable=lambda: param)
    MyManager.register('get_stop_signal', callable=lambda: stop_or_not)
    manager = MyManager(address=(args.ps_ip, 5000), authkey=b'queue')
    manager.start()
    
    q = manager.get_queue()  # queue for parameter_server signal process
    param_q = manager.get_param()  # init
    stop_signal = manager.get_stop_signal()  # stop

    init_myprocesses(this_rank, world_size, model,test_data,
                                                  q, param_q, stop_signal, run, args.backend)
    #p = Process(target=init_myprocesses, args=(this_rank, world_size, model,test_data,
    #                                               q, param_q, stop_signal, run, "gloo"))
    #p.start()
    #p.join()
    manager.shutdown()
