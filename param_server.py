# -*- coding: utf-8 -*-

from core.argParser import args
import os, shutil, pickle
import random, math
import numpy as np
import sys
import time
import datetime
import logging
from collections import deque
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

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForMaskedLM,
    AlbertTokenizer,
    get_linear_schedule_with_warmup,
)

from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import test_model
from utils.openImg import *
from utils.nlp import *

#device = torch.device(args.to_device)

logDir = os.getcwd() + "/../../models/"  + args.model + '/' + args.time_stamp + '/server/'
logFile = logDir + 'log'

def init_logging():
    global logDir

    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    with open(logFile, 'w') as fin:
        pass

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

staleness_file = '/tmp/staleness' + args.model + ".txt"

os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
os.environ['GLOO_SOCKET_IFNAME'] = 'vlan260'

device = None
deviceId = None
sampledClientSet = set()

# try to pick the right gpus
cudaPrefix = 'cuda:'
for i in range(4):
    try:
        device = torch.device(cudaPrefix+str(i))
        torch.cuda.set_device(i)
        deviceId = i
        logging.info(torch.rand(1).to(device=device))
        break
    except Exception as e:
        # no gpus available
        if i == 4:
            logging.info(e)
            deviceId = None
            logging.info('Turn to CPU device ...')
            device = 'cpu'
        else:
            continue

# os.environ['OMP_NUM_THREADS'] = args.threads
# os.environ['MKL_NUM_THREADS'] = args.threads

#torch.set_num_threads(int(args.threads))
#torch.cuda.set_device(args.gpu_device)


# initiate for nlp
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2') if args.task =='nlp' else None

def initiate_sampler_query(numOfClients):
    # Initiate the clientSampler 
    if args.sampler_path is None:
        clientSampler = ClientSampler(args.sample_mode, args.score_mode, filter=args.filter_less, sample_seed=args.sample_seed)
    else:
        # load sampler
        with open(args.sampler_path, 'rb') as loader:
            clientSampler = pickle.load(loader)

    collectedClients = 0
    initial_time = time.time()
    clientId = 1
    passed = False

    # In this simulation, we run data split on each worker, which amplifies the # of datasets
    # Waiting for the data information from clients, or timeout
    while collectedClients < numOfClients or (time.time() - initial_time) > 5000:
        if not queue.empty():
            tmp_dict = queue.get()

            # we only need to go over once
            if not passed and args.sampler_path is None:
                rank_src = list(tmp_dict.keys())[0]
                distanceVec = tmp_dict[rank_src][0]
                sizeVec = tmp_dict[rank_src][1]

                for index, dis in enumerate(distanceVec):
                    # since the worker rankId starts from 1, we also configure the initial dataId as 1
                    clientSampler.registerClient(rank_src, clientId, dis, sizeVec[index])
                    clientId += 1

                passed = True

            collectedClients += 1

    return clientSampler


def init_myprocesses(rank, size, model, test_data, queue, param_q, stop_signal, fn, backend):
    global sampledClientSet

    dist.init_process_group(backend, rank=rank, world_size=size)

    # After collecting all data information, then decide the clientId to run
    workerRanks = [int(v) for v in str(args.learners).split('-')]
    clientSampler = initiate_sampler_query(len(workerRanks))
    
    clientIdsToRun = []
    for wrank in workerRanks:
        nextClientIdToRun = clientSampler.nextClientIdToRun(hostId=wrank)
        clientSampler.clientOnHost([nextClientIdToRun], wrank)
        clientIdsToRun.append([nextClientIdToRun])
        sampledClientSet.add(nextClientIdToRun)
    
    dist.broadcast(tensor=torch.tensor(clientIdsToRun, dtype=torch.int).to(device=device), src=0)

    # Start the PS service
    fn(model, test_data, queue, param_q, stop_signal, clientSampler)

def init_dataset():
    global tokenizer

    if args.data_set == 'Mnist':
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

        model = tormodels.__dict__[args.model](num_classes=596)

    elif args.data_set == 'blog':
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False) 
        test_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

        model = AlbertForMaskedLM.from_pretrained('/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/nlp/')

    else:
        print('DataSet must be {}!'.format(['Mnist', 'Cifar', 'openImg', 'blog']))
        sys.exit(-1)

    model = model.to(device=device)

    return model, train_dataset, test_dataset

def run(model, test_data, queue, param_q, stop_signal, clientSampler):
    global logDir, sampledClientSet

    logging.info("====PS: get in run()")

    f_staleness = open(staleness_file, 'w')
    
    modelDir = os.getcwd() + "/../../models/"  + args.model
    modelPath = modelDir+'/'+str(args.model)+'.pth.tar' if args.model_path is None else args.model_path
    # convert gradient tensor to numpy structure
    if args.load_model:
        try:
            if deviceId is not None:
                model.load_state_dict(torch.load(modelPath, map_location=lambda storage, loc: storage.cuda(deviceId)))
            else:
                model.load_state_dict(torch.load(modelPath, map_location=lambda storage, loc: storage))
            logging.info("====Load model successfully\n")
        except Exception as e:
            logging.info("====Error: Failed to load model due to {}\n".format(str(e)))
            pass

    _tmp = OrderedDict(map(lambda item: (item[0], item[1].cpu().numpy()), model.state_dict().items()))
    
    workers = [int(v) for v in str(args.learners).split('-')]

    for _ in workers:
        param_q.put(_tmp)

    print('Begin!')

    epoch_train_loss = 0
    data_size_epoch = 0   # len(train_data), one epoch
    epoch_count = 1
    global_virtual_clock = 0.
    last_global_virtual_clock = global_virtual_clock

    staleness = 0
    learner_staleness = {l: 0 for l in workers}
    learner_local_step = {l: 0 for l in workers}
    learner_cache_step = {l: 0 for l in workers}
    pendingWorkers = {}
    test_results = {}

    s_time = time.time()
    epoch_time = s_time

    # In SSP, the fast workers have to wait the slowest worker a given duration
    # The fast worker exceeding the duration will be pushed into the queue to wait
    global_update = 0
    received_updates = 0
    last_global_model = [param for param in pickle.loads(pickle.dumps(model)).parameters()]
    clientsLastEpoch = []

    clientInfoFile = logDir + 'clientInfoFile'
    # dump the client info
    with open(clientInfoFile, 'wb') as fout:
        pickle.dump(clientSampler.getClientsInfo(), fout)

    while True:
        if not queue.empty():
            try:
                handle_start = time.time()
                tmp_dict = queue.get()
                rank_src = list(tmp_dict.keys())[0]

                [iteration_loss, trained_size, isWorkerEnd, clientIds, speed, testRes, virtualClock] = \
                [tmp_dict[rank_src][i] for i in range(1, len(tmp_dict[rank_src]))]
                #clientSampler.registerSpeed(rank_src, clientId, speed)

                if isWorkerEnd:
                    logging.info("====Worker {} has completed all its data computation!".format(rank_src))
                    learner_staleness.pop(rank_src)
                    if (len(learner_staleness) == 0):
                        f_staleness.close()
                        stop_signal.put(1)
                        break
                    continue

                learner_local_step[rank_src] += 1

                handlerStart = time.time()
                delta_wss = tmp_dict[rank_src][0]
                clientsLastEpoch += clientIds
                ratioSample = 0

                logging.info("====Start to merge models")
                for i, clientId in enumerate(clientIds):
                    gradients = None
                    ranSamples = float(speed[i].split('_')[1])

                    epoch_train_loss += iteration_loss[i]
                    data_size_epoch += trained_size[i]

                    # fraction of total samples on this specific node 
                    ratioSample = clientSampler.getSampleRatio(clientId, rank_src, args.is_even_avg)
                    delta_ws = delta_wss[i]

                    isSelected = True if clientId in sampledClientSet else False
                    # apply the update into the global model if the client is involved
                    for idx, param in enumerate(model.parameters()):
                        model_weight = torch.from_numpy(delta_ws[idx]).to(device=device)

                        if isSelected:
                            if received_updates == 0:
                                param.data = model_weight * ratioSample
                            else:
                                param.data += model_weight * ratioSample

                        if gradients is None:
                            gradients = (model_weight - last_global_model[idx]).flatten()
                        else:
                            gradients = torch.cat((gradients, (model_weight - last_global_model[idx]).flatten()))

                    # register the score
                    if args.score_mode == "loss":
                        clientSampler.registerScore(clientId, iteration_loss[i], time_stamp=epoch_count)
                    elif args.score_mode == "norm":
                        clientSampler.registerScore(clientId, gradients.norm(2).data.item() * ranSamples, time_stamp=epoch_count)
                    else:
                        clientSampler.registerScore(clientId, (1.0 - clientSampler.getClient(clientId).distance), time_stamp=epoch_count)

                    if isSelected:
                        received_updates += 1
                        global_virtual_clock = max(global_virtual_clock, last_global_virtual_clock + virtualClock[i])

                logging.info("====Done handling rank {}, with ratio {}".format(rank_src, ratioSample))

                # aggregate the test results
                updateEpoch = testRes[-1]
                if updateEpoch not in test_results:
                    # [top_1, top_5, loss, total_size, # of collected ranks]
                    test_results[updateEpoch] = [0., 0., 0., 0., 0]

                if updateEpoch != -1:
                    for idx, c in enumerate(testRes[:-1]):
                        test_results[updateEpoch][idx] += c

                    test_results[updateEpoch][-1] += 1
                    # have collected all ranks
                    if test_results[updateEpoch][-1] == len(workers):
                        logging.info("====After aggregation in epoch: {}, virtual_clock: {}, top_1: {} % ({}), top_5: {} % ({}), test loss: {}"
                                .format(updateEpoch, global_virtual_clock, round(test_results[updateEpoch][0]/test_results[updateEpoch][3]*100.0, 4), 
                                test_results[updateEpoch][0], round(test_results[updateEpoch][1]/test_results[updateEpoch][3]*100.0, 4), 
                                test_results[updateEpoch][1], test_results[updateEpoch][2]/test_results[updateEpoch][3]))


                handlerDur = time.time() - handlerStart
                global_update += 1

                # get the current minimum local staleness_sum_epoch
                currentMinStep = min([learner_local_step[key] for key in learner_local_step.keys()])

                staleness += 1
                learner_staleness[rank_src] = staleness

                # if the worker is within the staleness, then continue w/ local cache and do nothing
                # Otherwise, block it 
                if learner_local_step[rank_src] >= args.stale_threshold + currentMinStep:
                    pendingWorkers[rank_src] = learner_local_step[rank_src]
                    # lock the worker
                    logging.info("Lock worker " + str(rank_src) + " with localStep " + str(pendingWorkers[rank_src]) +
                                            " , while globalStep is " + str(currentMinStep) + "\n")
                
                # if the local cache is too stale, then update it
                elif learner_cache_step[rank_src] < learner_local_step[rank_src] - args.stale_threshold:
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
                    received_updates = 0
                    epoch_count += 1

                    logging.info("====Epoch {} completes {} clients, sampled rewards are: \n {} \n=========="
                                .format(epoch_count, len(clientsLastEpoch), {x:clientSampler.getScore(0, x) for x in sorted(clientsLastEpoch)}))

                    clientsLastEpoch = []
                    send_start = time.time()

                    for idx, param in enumerate(model.parameters()):
                        dist.broadcast(tensor=(param.data.to(device=device)), src=0)

                    # resampling the clients if necessary
                    if epoch_count % args.resampling_interval == 0:
                        logging.info("====Start to sample for epoch {}, global virtualClock is {}".format(epoch_count, global_virtual_clock))
                        sampledClients = sorted(clientSampler.resampleClients(max(args.total_worker, len(workers)), cur_time=epoch_count))
                        logging.info("====Try to resample clients, and result is {}".format(sampledClients))
                        sampledClientSet = set(sampledClients)

                        # simulate the optimal
                        if args.run_all:
                            sampledClients = clientSampler.getAllClients()

                        allocateClientToWorker = {}
                        allocateClientDict = {rank:0 for rank in workers}

                        # for those data lakes < # of iters, we use round-bin for load balance
                        for c in sampledClients:
                            clientDataSize = clientSampler.getClientSize(c)
                            numOfBatches = int(math.ceil(clientDataSize/args.batch_size))

                            if numOfBatches > args.upload_epoch:
                                workerId = workers[(c-1)%len(workers)]
                            else:
                                # pick the one w/ the least load
                                workerId = sorted(allocateClientDict, key=allocateClientDict.get)[0]

                            if workerId not in allocateClientToWorker:
                                allocateClientToWorker[workerId] = []

                            allocateClientToWorker[workerId].append(c)
                            allocateClientDict[workerId] = allocateClientDict[workerId] + 1
                        
                        for w in allocateClientToWorker.keys():
                            clientSampler.clientOnHost(allocateClientToWorker[w], w)

                    clientIdsToRun = [currentMinStep]
                    clientsList = []

                    endIdx = 0

                    for worker in workers:
                        learner_cache_step[worker] = currentMinStep
                        endIdx += clientSampler.getClientLenOnHost(worker)
                        clientIdsToRun.append(endIdx)
                        clientsList += clientSampler.getCurrentClientIds(worker)
                        # remove from the pending workers
                        del pendingWorkers[worker]

                    dist.broadcast(tensor=torch.tensor(clientIdsToRun, dtype=torch.int).to(device=device), src=0)
                    dist.broadcast(tensor=torch.tensor(clientsList, dtype=torch.int).to(device=device), src=0)

                    if global_update % args.display_step == 0:
                        logging.info("Handle Wight {} | Send {}".format(handle_dur, time.time() - send_start))

                    # dump the model into file for backup
                    if epoch_count % args.dump_epoch == 0:
                        torch.save(model.state_dict(), logDir+'/'+str(args.model)+'_'+str(currentMinStep)+'.pth.tar')

                        # dump sampler
                        with open(logDir + '/sampler_' + str(currentMinStep), 'wb') as fout:
                            pickle.dump(clientSampler, fout)

                        logging.info("====Dump model and sampler successfully")

                    last_global_model = [param for param in pickle.loads(pickle.dumps(model)).parameters()]
                    last_global_virtual_clock = global_virtual_clock

                # The training stop
                if(epoch_count >= args.epochs):
                    f_staleness.close()
                    stop_signal.put(1)
                    logging.info('Epoch is done: {}'.format(epoch_count))
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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    with open(logFile, 'w') as f:
        pass
    # Control the global random
    manual_seed = args.this_rank
    setup_seed(manual_seed)

    model, train_dataset, test_dataset = init_dataset()

    test_data = DataLoader(test_dataset, batch_size=args.test_bsz, shuffle=True)

    print("====PS: finish loading test_data")
    world_size = len(str(args.learners).split('-')) + 1
        
    this_rank = args.this_rank

    queue = Queue()
    param = Queue()
    stop_or_not = Queue()

    BaseManager.register('get_queue', callable=lambda: queue)
    BaseManager.register('get_param', callable=lambda: param)
    BaseManager.register('get_stop_signal', callable=lambda: stop_or_not)
    manager = BaseManager(address=(args.ps_ip, args.manager_port), authkey=b'queue')
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
