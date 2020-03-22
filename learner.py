# -*- coding: utf-8 -*-
# Thanks to qihua Zhou

from core.argParser import args
import os, shutil, sys, time, datetime, logging, pickle
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import random, math, gc
import multiprocessing, threading
import numpy as np
import torch
import torch.distributed as dist
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as tormodels

from utils.openImg import *
from utils.divide_data import partition_dataset, select_dataset, DataPartitioner
from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import MySGD, test_model
from utils.crosslossprox import CrossEntropyLossProx

device = torch.device(args.to_device)
#torch.set_num_threads(int(args.threads))

logDir = os.getcwd() + "/../../models/" + args.model + '/' + args.time_stamp + '/learner/'
logFile = logDir + 'log_'+str(args.this_rank)

def init_logging():
    global logDir

    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    files = [logFile, '/tmp/sampleDistribution']
    for file in files:
        with open(file, "w") as fout:
            pass

init_logging()

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logFile, mode='a'),
                        logging.StreamHandler()
                    ])

entire_train_data = None
os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
os.environ['GLOO_SOCKET_IFNAME'] = 'vlan260'
# os.environ['OMP_NUM_THREADS'] = args.threads
# os.environ['MKL_NUM_THREADS'] = args.threads

# try to pick the right gpus
cudaPrefix = 'cuda:'
for i in range(4):
    try:
        device = torch.device(cudaPrefix+str(i))
        torch.cuda.set_device(i)
        logging.info(torch.rand(1).to(device=device))
        break
    except Exception as e:
        # no gpus available
        if i == 4:
            logging.info(e)
            sys.exit(-1)
        else:
            continue

world_size = 0
global_trainDB = None
lastGlobalModel = None
nextClientIds = None
global_data_iter = {}
global_client_profile = {}

workers = [int(v) for v in str(args.learners).split('-')]

logging.info("===== Experiment start =====")


def report_data_info(rank, queue, entire_train_data):
    global nextClientIds

    # report data information to the clientSampler master
    queue.put({
        rank: [entire_train_data.getDistance(), entire_train_data.getSize()]
    })

    clientIdToRun = torch.zeros([world_size - 1], dtype=torch.int).to(device=device)
    dist.broadcast(tensor=clientIdToRun, src=0)
    nextClientIds = [clientIdToRun[args.this_rank - 1].item()]

def init_myprocesses(rank, size, model,
                   train_dataset, test_dataset,
                   q, param_q, stop_flag,
                   fn, backend, client_cfg):
    print("====Worker: init_myprocesses")
    fn(rank, model, train_dataset, test_dataset, q, param_q, stop_flag, client_cfg)

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

        model = tormodels.__dict__[args.model](num_classes=596)
        
    else:
        print('DataSet must be {} or {}!'.format('Mnist', 'Cifar'))
        sys.exit(-1)

    model = model.to(device=device)

    # initiate the device information - normalized computation speed by enforcing sleeping, bandwidth
    client_cfg = {}

    if os.path.exists(args.client_path):
        with open(args.client_path, 'r') as fin:
            for line in fin.readlines():
                items = line.strip().split()
                clientId, compute, commu = int(items[0]), float(items[1]), float(items[2])
                global_client_profile[clientId] = [compute, commu]

    return model, train_dataset, test_dataset, client_cfg

def run_forward_pass(model, test_data, criterion=nn.NLLLoss()):
    test_loss = 0.
    test_len = 0.

    model.eval()

    for data, target in test_data:
        data, target = Variable(data).cuda(), Variable(target).cuda()
        output = model(data)
        test_loss += criterion(output, target).data.item()
        test_len += len(target)
        
    # loss function averages over batch size
    test_loss /= float(len(test_data))

    return test_loss

def run_client(clientId, model, criterion, iters, learning_rate, argdicts = {}):
    global global_trainDB, global_data_iter, lastGlobalModel

    curBatch = -1
    optimizer = MySGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    train_data_itr_list = []

    if clientId not in global_data_iter:
        client_train_data = select_dataset(clientId, global_trainDB, batch_size=args.batch_size)
        train_data_itr = iter(client_train_data)
        total_batch_size = len(train_data_itr)
        global_data_iter[clientId] = [train_data_itr, curBatch, total_batch_size]
    else:
        [train_data_itr, curBatch, total_batch_size] = global_data_iter[clientId]

    local_trained = 0
    numOfPreWarmUp = 1
    epoch_train_loss = 0.
    comp_duration = 0.
    norm_gradient = 0.
    count = 0

    train_data_itr_list.append(train_data_itr)
    run_start = time.time()

    numOfFailures = 0
    numOfTries = 5
    model.train()

    for itr in range(iters):
        it_start = time.time()

        fetchSuccess = False
        while not fetchSuccess and numOfFailures < numOfTries:
            try:
                try:
                    (data, target) = next(train_data_itr_list[0])
                    fetchSuccess = True
                except Exception:
                    try:
                        train_data_itr_list[0]._shutdown_workers()
                        del train_data_itr_list[0]
                    except Exception as e:
                        logging.info("====Error {}".format(e))

                    # reload data after finishing the epoch
                    if len(train_data_itr_list) == 0:
                        for i in range(numOfPreWarmUp):
                            train_data_itr_list.append(iter(select_dataset(clientId, global_trainDB, batch_size=args.batch_size)))
                    else:
                        train_data_itr_list.append(iter(select_dataset(clientId, global_trainDB, batch_size=args.batch_size)))

                    (data, target) = next(train_data_itr_list[0])
                    fetchSuccess = True
            except Exception as e:
                numOfFailures += 1
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))
                time.sleep(0.5)

        if numOfFailures >= numOfTries:
            break

        numOfFailures = 0
        curBatch = curBatch + 1

        data, target = Variable(data).to(device=device), Variable(target).to(device=device)
        local_trained += len(target)

        optimizer.zero_grad()

        comp_start = time.time()
        output = model(data)

        if args.proxy_avg:
            loss = criterion(output, target, global_weight=lastGlobalModel.parameters(), 
                            individual_weight=model.parameters(), mu=0.01)
        else:
            loss = criterion(output, target)

        loss.backward()
        delta_w = optimizer.get_delta_w(learning_rate)

        #if itr < total_batch_size:
        epoch_train_loss += (loss.data.item() * len(target))
        count += len(target)
        
        for idx, param in enumerate(model.parameters()):
            param.data -= delta_w[idx].to(device=device)
            #norm_gradient += delta_w[idx].norm(2).to(device=device)

        comp_duration = (time.time() - comp_start)
    
        logging.info('For client {}, upload iter {}, epoch {}, Batch {}/{}, Loss:{} | TotalTime {} | Comptime: {} \n'
                    .format(clientId, argdicts['iters'], int(curBatch/total_batch_size),
                    (curBatch % total_batch_size), total_batch_size, round(loss.data.item(), 4), 
                    round(time.time() - it_start, 4), round(comp_duration, 4)))

    # save the state of this client if # of batches > iters, since we want to pass over all samples at least one time
    if total_batch_size > iters and len(train_data_itr_list) > 0:
        global_data_iter[clientId] = [train_data_itr_list[0], curBatch, total_batch_size]
    else:
        for loader in train_data_itr_list:
            loader._shutdown_workers()
        del train_data_itr_list
        del global_data_iter[clientId]

    model_param = [param.data.cpu().numpy() for param in model.parameters()]
    
    time_spent = time.time() - run_start

    # add bias to the virtual clock, computation x (# of trained samples) + communication
    if clientId in global_client_profile:
        time_cost = global_client_profile[clientId][0] * count + global_client_profile[clientId][1]
    else:
        time_cost = time_spent

    speed = None
    if count > 0:
        speed = time_spent/float(count) 
        epoch_train_loss /= float(count)
    else:
        logging.info("====Failed to run client {}".format(clientId))

    return model_param, epoch_train_loss, local_trained, speed, time_cost

def run(rank, model, train_data, test_data, queue, param_q, stop_flag, client_cfg):
    print("====Worker: Start running")

    global nextClientIds, global_trainDB, lastGlobalModel

    global_trainDB = train_data
    startTime = time.time()

    # Fetch the initial parameters from the server side (we called it parameter_server)
    while True:
        if not param_q.empty():
            param_dict = param_q.get()
            tmp = OrderedDict(map(lambda item: (item[0], torch.from_numpy(item[1])),
                                  param_dict.items()))
            model.load_state_dict(tmp)
            break

    lastGlobalModel = model
    criterion = CrossEntropyLossProx().to(device=device) if args.proxy_avg else torch.nn.CrossEntropyLoss().to(device=device)

    print('Begin!')
    logging.info('\n' + repr(args) + '\n')

    logDir = "/tmp/" + args.model + '_' + str(args.this_rank)
    if os.path.isdir(logDir):
        shutil.rmtree(logDir)
    os.mkdir(logDir)

    learning_rate = args.learning_rate
    uploadEpoch = -1
    testResults = [0, 0, 0, len(test_data)]
    last_test = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        try:
            if epoch % args.decay_epoch == 0:
                learning_rate = max(1e-5, learning_rate * args.decay_factor)

            trainedModels = []
            preTrainedLoss = []
            trainedSize = []
            trainSpeed = []
            virtualClock = []
            ranClients = []

            computeStart = time.time()
            for nextClientId in nextClientIds:
                if args.forward_pass:
                    forward_dataset = select_dataset(nextClientId, global_trainDB, batch_size=args.test_bsz)
                    forward_loss = run_forward_pass(model, forward_dataset, criterion=criterion)

                _model_param, _loss, _trained_size, _speed, _time = run_client(clientId=nextClientId, 
                        model=pickle.loads(pickle.dumps(lastGlobalModel)), learning_rate=learning_rate, 
                        criterion=criterion, iters=args.upload_epoch,
                        argdicts={'iters': epoch})

                if _speed is None:
                    continue

                trainedModels.append(_model_param)
                preTrainedLoss.append(_loss if not args.forward_pass else forward_loss)
                trainedSize.append(_trained_size)
                trainSpeed.append(_speed)
                virtualClock.append(_time)
                ranClients.append(nextClientId)

                gc.collect()

            computeEnd = time.time() - computeStart

            # upload the weight
            sendStart = time.time()
            testResults.append(uploadEpoch)
            queue.put_nowait({rank: [trainedModels, preTrainedLoss, trainedSize, False, ranClients, trainSpeed, testResults, virtualClock]})
            uploadEpoch = -1
            sendDur = time.time() - sendStart

            # wait for new models
            receStart = time.time()

            for idx, param in enumerate(model.parameters()):
                dist.broadcast(tensor=param.data, src=0)

            # receive current minimum step, and the clientIdLen for next training
            step_tensor = torch.zeros([world_size], dtype=torch.int).to(device=device)
            dist.broadcast(tensor=step_tensor, src=0)
            globalMinStep = step_tensor[0].item()
            totalLen = step_tensor[-1].item()
            endIdx = step_tensor[args.this_rank].item()
            startIdx = 0 if args.this_rank == 1 else step_tensor[args.this_rank - 1].item()

            clients_tensor = torch.zeros([totalLen], dtype=torch.int).to(device=device)
            dist.broadcast(tensor=clients_tensor, src=0)
            nextClientIds = [clients_tensor[x].item() for x in range(startIdx, endIdx)]

            receDur = time.time() - receStart
            # If we simulate multiple workers, we have to do deep copy
            lastGlobalModel = model

            evalStart = time.time()
            # test the model if necessary
            if epoch % int(args.eval_interval) == 0:
                test_loss, acc, acc_5, testResults = test_model(rank, model, test_data, criterion=criterion)
                logging.info("After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {}, test_5_accuracy {} \n"
                            .format(epoch, round(time.time() - startTime, 4), round(time.time() - evalStart, 4), test_loss, acc, acc_5))

                uploadEpoch = epoch
                last_test = time.time()

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))
            break

        if time.time() - last_test > args.test_interval:
            last_test = time.time()
            test_loss, acc = test_model(rank, model, test_data, criterion=criterion)
            
            logging.info("For epoch {}, CumulTime {}, test_loss {}, test_accuracy {} \n"
                .format(epoch, time.time() - startTime, test_loss, acc))

            last_test = time.time()

        if stop_flag.value:
            break

    queue.put({rank: [None, None, None, True, -1, -1]})
    logging.info("Worker {} has completed epoch {}!".format(args.this_rank, epoch))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(args.this_rank)

    train_bsz = args.batch_size

    model, train_dataset, test_dataset, client_cfg = init_dataset()

    splitTrainRatio = []

    # Initialize PS - client communication channel
    world_size = len(workers) + 1
    this_rank = args.this_rank

    BaseManager.register('get_queue')
    BaseManager.register('get_param')
    BaseManager.register('get_stop_signal')
    manager = BaseManager(address=(args.ps_ip, args.manager_port), authkey=b'queue')
    manager.connect()

    q = manager.get_queue()  # queue for parameter_server signal process
    param_q = manager.get_param()  # init
    stop_signal = manager.get_stop_signal()  # stop
    dist.init_process_group(args.backend, rank=this_rank, world_size=world_size)

    # Split the dataset
    # total_worker != 0 indicates we create more virtual clients for simulation
    if args.total_worker > 0 and args.duplicate_data == 1:
        workers = [i for i in range(1, args.total_worker + 1)]

    # load data partitioner (entire_train_data)
    dataConf = os.path.join(args.data_dir, 'sampleConf') if args.data_set == 'imagenet' else None

    entire_train_data = DataPartitioner(data=train_dataset, splitConfFile=dataConf, 
                        numOfClass=args.num_class, dataMapFile=args.data_mapfile)

    dataDistribution = [int(x) for x in args.sequential.split('-')]
    distributionParam = [float(x) for x in args.zipf_alpha.split('-')]

    for i in range(args.duplicate_data):
        partition_dataset(entire_train_data, workers, splitTrainRatio, dataDistribution[i], 
                                    filter_class=args.filter_class, arg = {'balanced_client':0, 'param': distributionParam[i]})
    entire_train_data.log_selection()

    report_data_info(this_rank, q, entire_train_data)
    splitTestRatio = []

    testsetPartitioner = DataPartitioner(data=test_dataset, isTest=True, numOfClass=args.num_class)
    partition_dataset(testsetPartitioner, [i for i in range(world_size-1)], splitTestRatio)
    test_data = select_dataset(this_rank, testsetPartitioner, batch_size=args.test_bsz, isTest=True)

    stop_flag = Value(c_bool, False)
    init_myprocesses(this_rank, world_size, model, entire_train_data, test_data,
                                          q, param_q, stop_flag,
                                          run, args.backend, client_cfg)
