# -*- coding: utf-8 -*-

from core.argParser import args
import os, shutil
import random
import sys
import time
import datetime
import logging
from clientSampler import ClientSampler
from collections import OrderedDict
from multiprocessing.managers import BaseManager
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as tormodels

from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import test_model
from utils.openImg import *

device = torch.device(args.to_device)

logFile = '/tmp/torch/log_' + str(datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S'))

def init_logging():
    if not os.path.isdir('/tmp/torch'):
        os.mkdir('/tmp/torch')

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[
                            logging.FileHandler(logFile, mode='a'),
                            logging.StreamHandler()
                        ])

init_logging()

entire_train_data = None
sample_size_dic = {}

trainloss_file = '/tmp/trainloss' + args.model + '.txt'
staleness_file = '/tmp/staleness' + args.model + ".txt"
os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = str(int(args.ps_port) + 10*int(args.gpu_device))
#os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
os.environ['GLOO_SOCKET_IFNAME'] = 'vlan260'

# os.environ['OMP_NUM_THREADS'] = args.threads
# os.environ['MKL_NUM_THREADS'] = args.threads

#torch.set_num_threads(int(args.threads))
#torch.cuda.set_device(args.gpu_device)

def initiate_sampler_query(numOfClients):
    # Initiate the clientSampler
    clientSampler = ClientSampler(args.sample_mode, args.score_mode)
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

    # convert gradient tensor to numpy structure
    if args.load_model:
        try:
            model.load_state_dict(torch.load(logDir+'/'+str(args.model)+'.pth.tar'))
            logging.info("====Load model successfully\n")
        except Exception as e:
            logging.info("====Error: Failed to load model due to {}\n".format(str(e)))
            pass

    if not os.path.isdir(logDir):
        os.mkdir(logDir)

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
    newEpoch = 0

    logging.info(repr(clientSampler.getClientsInfo()))

    while True:
        if not queue.empty():
            try:
                handle_start = time.time()
                tmp_dict = queue.get()
                rank_src = list(tmp_dict.keys())[0]

                [iteration_loss, trained_size, isWorkerEnd, clientId, speed] = [tmp_dict[rank_src][i] for i in range(1, len(tmp_dict[rank_src]))]
                #clientSampler.registerSpeed(rank_src, clientId, speed)
                if args.score_mode == "loss":
                    clientSampler.registerScore(clientId, iteration_loss)
                else:
                    sc = 1.0 - clientSampler.getScore(rank_src, clientId)
                    clientSampler.registerScore(clientId, sc)

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
                        param.data -= (torch.from_numpy(delta_ws[idx]).to(device=device))
                    else:
                        if newEpoch == 0:
                            param.data = (torch.from_numpy(delta_ws[idx]).to(device=device)) * ratioSample
                        else:
                            param.data += (torch.from_numpy(delta_ws[idx]).to(device=device)) * ratioSample

                newEpoch += 1

                handlerDur = time.time() - handlerStart
                global_update += 1
                currentMinStep = 9999999999

                # get the current minimum local staleness_sum_epoch
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
                    workersToSend = sorted(workersToSend)
                    send_start = time.time()

                    for idx, param in enumerate(model.parameters()):
                        dist.broadcast(tensor=(param.data.to(device=device)), src=0)

                    # resampling the clients if necessary
                    if epoch_count % args.resampling_interval == 0:
                        sampledClients = sorted(clientSampler.resampleClients(len(workers), max(args.total_worker, len(workers))))
                        logging.info("====Try to resample clients with metrics {}, result is {}".format(repr(clientSampler.getAllMetrics()), repr(sampledClients)))
                        for i, w in enumerate(workers):
                            clientSampler.clientOnHost(sampledClients[i], w)

                    clientIdsToRun = [currentMinStep]
                    for worker in workersToSend:
                        learner_cache_step[worker] = currentMinStep
                        clientIdsToRun.append(clientSampler.getCurrentClientId(worker))
                        # remove from the pending workers
                        del pendingWorkers[worker]

                    dist.broadcast(tensor=torch.tensor(clientIdsToRun, dtype=torch.int).to(device=device), src=0)

                    newEpoch = 0

                    if global_update % args.display_step == 0:
                        logging.info("Handle Wight {} | Send {}".format(handle_dur, time.time() - send_start))

                    # dump the model into file for backup
                    if epoch_count % args.dump_epoch == 0:
                        torch.save(model.state_dict(), logDir+'/'+str(args.model)+'_'+str(currentMinStep)+'.pth.tar')
                        logging.info("====Dump model successfully")

                # The training stop
                if(epoch_count >= args.epochs * args.upload_epoch):
                    f_staleness.close()
                    stop_signal.put(1)
                    print('Epoch is done: {}'.format(epoch_count))
                    break

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("====Error: " + str(e) + '\n')
                logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))

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

    clientIdsToRun = []
    for wrank in workerRanks:
        nextClientIdToRun = clientSampler.nextClientIdToRun(hostId=wrank)
        clientSampler.clientOnHost(nextClientIdToRun, wrank)
        clientIdsToRun.append(nextClientIdToRun)

    dist.broadcast(tensor=torch.tensor(clientIdsToRun, dtype=torch.int).to(device=device), src=0)

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

        model = tormodels.__dict__[args.model](num_classes=10)

    elif args.data_set == 'cifar10':
        train_transform, test_transform = get_data_transform('cifar')
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                        transform=test_transform)

        model = tormodels.__dict__[args.model](num_classes=10)

    elif args.data_set == "imagenet":
        train_transform, test_transform = get_data_transform('imagenet')
        train_dataset = datasets.ImageNet(args.data_dir, split='train', download=False, transform=train_transform)
        test_dataset = datasets.ImageNet(args.data_dir, split='val', download=False, transform=test_transform)

        model = tormodels.__dict__[args.model]()

    elif args.data_set == 'emnist':
        test_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=False, download=True, transform=transforms.ToTensor())
        train_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=True, download=True, transform=transforms.ToTensor())

        model = tormodels.__dict__[args.model](num_classes=47)

    elif args.data_set == 'openImg':
        train_transform, test_transform = get_data_transform('openImg')
        train_dataset = OPENIMG(args.data_dir, train=True, transform=train_transform)
        test_dataset = OPENIMG(args.data_dir, train=False, transform=test_transform)

        model = tormodels.__dict__[args.model](num_classes=600)

    else:
        print('DataSet must be {} or {}!'.format('Mnist', 'Cifar'))
        sys.exit(-1)

    model = model.to(device=device)

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

    test_data = DataLoader(test_dataset, batch_size=args.test_bsz, shuffle=True)

    print("====PS: finish loading test_data")
    world_size = len(str(args.learners).split('-')) + 1

    this_rank = args.this_rank

    queue = Queue()
    param = Queue()
    stop_or_not = Queue()

    MyManager.register('get_queue', callable=lambda: queue)
    MyManager.register('get_param', callable=lambda: param)
    MyManager.register('get_stop_signal', callable=lambda: stop_or_not)
    manager = MyManager(address=(args.ps_ip, args.manager_port+10*int(args.gpu_device)), authkey=b'queue')
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
