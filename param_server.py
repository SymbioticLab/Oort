# -*- coding: utf-8 -*-

from core.argParser import args
import os, shutil, pickle, gc, json
import random, math
import numpy as np
import sys, socket
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
from torch_baidu_ctc import CTCLoss

from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import test_model
from utils.openImg import *
from utils.nlp import *
from utils.inception import *
from utils.stackoverflow import *
from utils.yogi import *
from utils.transforms_wav import *
from utils.transforms_stft import *
from utils.speech import *
from utils.resnet_speech import *

# for voice
from utils.voice_model import DeepSpeech, supported_rnns

#device = torch.device(args.to_device)

logDir = os.getcwd() + "/../../models/"  + args.model + '/' + args.time_stamp + '/server/'
logFile = logDir + 'log'
modelDir = os.getcwd() + "/../../models/"  + args.model
modelPath = modelDir+'/'+str(args.model)+'.pth.tar' if args.model_path is None else args.model_path


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

def dump_ps_ip():
    hostname_map = {}
    with open('ipmapping', 'rb') as fin:
        hostname_map = pickle.load(fin)

    ps_ip = str(hostname_map[str(socket.gethostname())])
    args.ps_ip = ps_ip

    with open(logDir+'ip', 'wb') as fout:
        pickle.dump(ps_ip, fout)

init_logging()
dump_ps_ip()

entire_train_data = None
sample_size_dic = {}

staleness_file = logDir + 'staleness.txt'

os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
os.environ['GLOO_SOCKET_IFNAME'] = 'vlan260'
os.environ['NCCL_DEBUG'] = 'INFO'

# cudaPrefix = 'cuda:'
# logging.info("====CUDA_VISIBLE_DEVICES is {}, {}".format(os.environ["CUDA_VISIBLE_DEVICES"], torch.cuda.device_count()))

# deviceId = 0#int(os.environ["CUDA_VISIBLE_DEVICES"])
# #torch.cuda.set_device(deviceId)
# device = cudaPrefix+str(deviceId) #torch.device(0)

# logging.info("====Pick gpu {}, {}".format(torch.cuda.current_device(), device))
# logging.info(f"====NCCL config ip: {args.ps_ip}, port: {args.ps_port}")

device = None
deviceId = None
sampledClientSet = set()

# try to pick the right gpus
cudaPrefix = 'cuda:'
for i in range(3, -1, -1):
    try:
        device = torch.device(cudaPrefix+str(i))
        torch.cuda.set_device(i)
        deviceId = i
        logging.info(torch.rand(1).to(device=device))
        break
    except Exception as e:
        # no gpus available
        if i == 0:
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

sampledClientSet = set()
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

    # load client profiles
    global_client_profile = {}
    if os.path.exists(args.client_path):
        with open(args.client_path, 'rb') as fin:
            # {clientId: [computer, bandwidth]}
            global_client_profile = pickle.load(fin)

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
                    systemProfile = global_client_profile[clientId] if clientId in global_client_profile else [1.0, 1.0]
                    clientSampler.registerClient(rank_src, clientId, dis, sizeVec[index], speed=systemProfile)
                    clientSampler.registerDuration(clientId, 
                        batch_size=args.batch_size, upload_epoch=args.upload_epoch, 
                        model_size=args.model_size)
                    
                    clientId += 1

                passed = True

            collectedClients += 1

    logging.info("====Info of all feasible clients {}".format(clientSampler.getDataInfo()))

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
    
    clientTensor = torch.tensor(clientIdsToRun, dtype=torch.int, device=device)
    dist.broadcast(tensor=clientTensor, src=0)

    # Start the PS service
    fn(model, test_data, queue, param_q, stop_signal, clientSampler)

def init_dataset():
    global tokenizer

    outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47, 'openImg': 596, 'google_speech': 35, 'femnist': 62}

    logging.info("====Initialize the model")

    if args.task == 'nlp':
        # we should train from scratch
        config = AutoConfig.from_pretrained(os.path.join(args.data_dir, 'albert-base-v2-config.json'))
        model = AutoModelWithLMHead.from_config(config)
    elif args.task == 'tag-one-sample':
        # Load LR model for tag prediction
        model = LogisticRegression(args.vocab_token_size, args.vocab_tag_size)
    elif args.task == 'speech':

        if args.model == 'mobilenet':
            model = mobilenet_v2(num_classes=outputClass[args.data_set], inchannels=1)
        elif args.model == "resnet18":
            model = resnet18(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet34":
            model = resnet34(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet50":
            model = resnet50(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet101":
            model = resnet101(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet152":
            model = resnet152(num_classes=outputClass[args.data_set], in_channels=1)
        else:
            # Should not reach here
            print('Model must be resnet or mobilenet')
            sys.exit(-1)
    elif args.task == 'voice':
        # Initialise new model training
        with open(args.labels_path) as label_file:
            labels = json.load(label_file)

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[args.rnn_type.lower()],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional)
    else:
        if args.model == 'mnasnet':
            model = MnasNet(num_classes=outputClass[args.data_set])
        elif args.model == "lr":
            model = LogisticRegression(args.input_dim, outputClass[args.data_set])
        else:
            model = tormodels.__dict__[args.model](num_classes=outputClass[args.data_set])

    if args.load_model:
        try:
            with open(modelPath, 'rb') as fin:
                model = pickle.load(fin)

            #model.load_state_dict(torch.load(modelPath, map_location=lambda storage, loc: storage.cuda(deviceId)))
            logging.info("====Load model successfully\n")
        except Exception as e:
            logging.info("====Error: Failed to load model due to {}\n".format(str(e)))
            sys.exit(-1)

    model = model.to(device=device)
    model.eval()

    logging.info("====Finish loading model")

    return model, [], []

def run(model, test_data, queue, param_q, stop_signal, clientSampler):
    global logDir, sampledClientSet

    logging.info("====PS: get in run()")

    f_staleness = open(staleness_file, 'w')
    
    #if not args.load_model:
    for name, param in model.named_parameters():
        dist.broadcast(tensor=param.data.to(device=device), src=0)
        logging.info(f"====Model parameters name: {name}")

    workers = [int(v) for v in str(args.learners).split('-')]

    print('Begin!')

    epoch_train_loss = 0
    data_size_epoch = 0   # len(train_data), one epoch
    epoch_count = 1
    global_virtual_clock = 0.
    round_duration = 0.

    staleness = 0
    learner_staleness = {l: 0 for l in workers}
    learner_local_step = {l: 0 for l in workers}
    learner_cache_step = {l: 0 for l in workers}
    pendingWorkers = {}
    test_results = {}
    virtualClientClock = {}
    exploredPendingWorkers = []
    avgUtilLastEpoch = 0.

    s_time = time.time()
    epoch_time = s_time

    global_update = 0
    received_updates = 0
    last_global_model = [param for param in pickle.loads(pickle.dumps(model)).parameters()]
    clientsLastEpoch = []
    sumDeltaWeights = []
    clientWeightsCache = {}
    last_sampled_clients = None

    gradient_controller = None
    # initiate yogi if necessary
    if args.gradient_policy == 'yogi':
        gradient_controller = YoGi(eta=args.yogi_eta, tau=args.yogi_tau, beta=args.yogi_beta, beta2=args.yogi_beta2)

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

                if not args.test_only or epoch_count == 1:
                    for i, clientId in enumerate(clientIds):
                        gradients = None
                        ranSamples = float(speed[i].split('_')[1])

                        data_size_epoch += trained_size[i]

                        # fraction of total samples on this specific node 
                        ratioSample = clientSampler.getSampleRatio(clientId, rank_src, args.is_even_avg)
                        delta_ws = delta_wss[i]
                        #clientWeightsCache[clientId] = [torch.from_numpy(x).to(device=device) for x in delta_ws]

                        epoch_train_loss += ratioSample * iteration_loss[i]
                        isSelected = True if clientId in sampledClientSet else False
                        # apply the update into the global model if the client is involved
                        for idx, param in enumerate(model.parameters()):
                            model_weight = torch.from_numpy(delta_ws[idx]).to(device=device)

                            # model_weight is the delta of last model
                            if isSelected:
                                # the first received client
                                if received_updates == 0:
                                    sumDeltaWeights.append(model_weight * ratioSample)
                                else:
                                    sumDeltaWeights[idx] += model_weight * ratioSample

                        # bias term for global speed
                        virtual_c = virtualClientClock[clientId] if clientId in virtualClientClock else 1.
                        clientUtility = 1. 

                        size_of_sample_bin = 1. 

                        if args.capacity_bin == True:
                            size_of_sample_bin = min(clientSampler.getClient(clientId).size, args.upload_epoch*args.batch_size)

                        # register the score
                        if args.score_mode == "loss":
                            clientUtility = math.sqrt(iteration_loss[i]) * size_of_sample_bin
                        elif args.score_mode == "norm":
                            clientUtility = math.sqrt(iteration_loss[i]) * size_of_sample_bin
                        elif args.score_mode == "size":
                            clientUtility = size_of_sample_bin
                        else:
                            clientUtility = (1.0 - clientSampler.getClient(clientId).distance)
                            
                        clientSampler.registerScore(clientId, clientUtility, auxi=math.sqrt(iteration_loss[i]),
                                                    time_stamp=epoch_count, duration=virtual_c
                                      )
                        if isSelected:
                            received_updates += 1

                        avgUtilLastEpoch += ratioSample * clientUtility

                        delta_ws = None
                        del delta_ws

                logging.info("====Done handling rank {}, with ratio {}, now collected {} clients".format(rank_src, ratioSample, received_updates))

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
                        top_1_str = 'top_1: '
                        top_5_str = 'top_5: '

                        if args.task == 'tag':
                            top_1_str = 'all-or-nothing: '
                            top_5_str = 'accuracy: '

                        try:
                            logging.info("====After aggregation in epoch: {}, virtual_clock: {}, {}: {} % ({}), {}: {} % ({}), test loss: {}, test len: {}"
                                    .format(updateEpoch, global_virtual_clock, top_1_str, round(test_results[updateEpoch][0]/test_results[updateEpoch][3]*100.0, 4), 
                                    test_results[updateEpoch][0], top_5_str, round(test_results[updateEpoch][1]/test_results[updateEpoch][3]*100.0, 4), 
                                    test_results[updateEpoch][1], test_results[updateEpoch][2]/test_results[updateEpoch][3], test_results[updateEpoch][3]))
                        except Exception as e:
                            logging.info(f"====Error {e}")

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

                tmp_dict = None
                delta_wss = None

                del delta_wss
                del tmp_dict

                if len(workersToSend) > 0:
                    # assign avg reward to explored, but not ran workers
                    for clientId in exploredPendingWorkers:
                        assert (avgUtilLastEpoch > 0)
                        clientSampler.registerScore(clientId, avgUtilLastEpoch,
                                                time_stamp=epoch_count, duration=virtualClientClock[clientId],
                                                success=False
                                  )

                    workersToSend = sorted(workersToSend)
                    epoch_count += 1
                    avgUtilLastEpoch = 0.

                    logging.info("====Epoch {} completes {} clients with loss {}, sampled rewards are: \n {} \n=========="
                                .format(epoch_count, len(clientsLastEpoch), epoch_train_loss, {x:clientSampler.getScore(0, x) for x in sorted(clientsLastEpoch)}))

                    epoch_train_loss = 0.
                    clientsLastEpoch = []
                    send_start = time.time()

                    # transformation of gradients if necessary
                    if gradient_controller is not None:
                        sumDeltaWeights = gradient_controller.update(sumDeltaWeights)

                    for idx, param in enumerate(model.parameters()):
                        if not args.test_only:
                            param.data += sumDeltaWeights[idx]
                        dist.broadcast(tensor=(param.data.to(device=device)), src=0)

                    # resampling the clients if necessary
                    if epoch_count % args.resampling_interval == 0 or epoch_count == 2:
                        logging.info("====Start to sample for epoch {}, global virtualClock: {}, round_duration: {}"
                                        .format(epoch_count, global_virtual_clock, round_duration))
                        
                        numToRealRun = max(args.total_worker, len(workers))
                        numToSample = int(numToRealRun * args.overcommit)
                        sampledClientsRealTemp = last_sampled_clients if args.fixed_clients and last_sampled_clients else sorted(clientSampler.resampleClients(numToSample, cur_time=epoch_count))

                        sampledClientsReal = []
                        for virtualClient in sampledClientsRealTemp:
                            roundDuration = clientSampler.getCompletionTime(virtualClient, 
                                                    batch_size=args.batch_size, upload_epoch=args.upload_epoch, 
                                                    model_size=args.model_size)

                            if clientSampler.isClientActive(virtualClient, roundDuration + global_virtual_clock):
                                sampledClientsReal.append(virtualClient)

                        last_sampled_clients = sampledClientsReal

                        # we decide to simulate the wall time and remove 1. stragglers 2. off-line
                        completionTimes = []
                        virtualClientClock = {}
                        for virtualClient in sampledClientsReal:
                            roundDuration = clientSampler.getCompletionTime(virtualClient, 
                                                    batch_size=args.batch_size, upload_epoch=args.upload_epoch, 
                                                    model_size=args.model_size)
                            completionTimes.append(roundDuration)
                            virtualClientClock[virtualClient] = roundDuration

                        # get the top-k completions
                        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
                        top_k_index = sortedWorkersByCompletion[:numToRealRun]
                        sampledClients = [sampledClientsReal[k] for k in top_k_index]

                        #if args.test_only and epoch_count == 2:
                        #    sampledClients = clientSampler.getAllClients()

                        exploredPendingWorkers = [sampledClientsReal[k] for k in sortedWorkersByCompletion[numToRealRun:]]
                        sampledClientSet = set(sampledClients)
                        round_duration = completionTimes[top_k_index[-1]]

                        logging.info("====Try to resample clients, and result is: \n{}\n while final takes: \n {} \n virtual duration is {}, {} clients become offline"
                                    .format(sampledClientsReal, sampledClients, virtualClientClock, len(sampledClientsRealTemp) - len(sampledClientsReal)))

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
                    # if epoch_count % args.dump_epoch == 0:
                    #     torch.save(model.state_dict(), logDir+'/'+str(args.model)+'_'+str(currentMinStep)+'.pth.tar')
                    #     # with open(logDir+'/'+str(args.model)+'_'+str(currentMinStep)+'.pth.tar', 'wb') as fout:
                    #     #     pickle.dump(model.to(device='cpu'), fout)

                    #     # dump sampler
                    #     # with open(logDir + '/sampler_' + str(currentMinStep), 'wb') as fout:
                    #     #     pickle.dump(clientSampler, fout)

                    #     # # dump metrics
                    #     # with open(logDir + '/sampler_metrics_' + str(currentMinStep), 'wb') as fout:
                    #     #     pickle.dump(clientSampler.getAllMetrics(), fout)

                    #     logging.info("====Dump model and sampler successfully")
                    #     model = model.to(device=device)

                    last_global_model = [param for param in pickle.loads(pickle.dumps(model)).parameters()]
                    
                    # update the virtual clock
                    global_virtual_clock += round_duration
                    received_updates = 0

                    # clientKeys = sorted(clientWeightsCache.keys())
                    
                    # # calculate the L2-norm of weights
                    # tempSumL2Norm = [w.norm(2).item() for w in sumDeltaWeights]
                    # assert(len(tempSumL2Norm) == len(clientWeightsCache[clientKeys[0]]))
                    
                    # logging.info(f"====Epoch: {epoch_count}, Avg L2-norm is: {tempSumL2Norm}")

                    # tempModelL2Norm = [param.data.norm(2).item() for param in model.parameters()]
                    # logging.info(f"====Model L2-norm is: {tempModelL2Norm}")

                    # for clientId in clientKeys:
                    #     weights = clientWeightsCache[clientId]
                    #     tempL2Norm = []
                    #     for pIdx, weight in enumerate(weights):
                    #         tempL2Norm.append((weight - sumDeltaWeights[pIdx]).norm(2).item())

                    #     logging.info(f"====For clientId {clientId}, L2-norm is: {tempL2Norm}")

                    sumDeltaWeights = []
                    clientWeightsCache = {}

                    gc.collect()

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
    setup_seed(args.this_rank)

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

    logging.info("====Start to initialize dataset")

    model, train_dataset, test_dataset = init_dataset()
    logging.info("====Len of train_dataset: {}, Len of test_dataset: {}".format(len(train_dataset), len(test_dataset)))

    test_data = []#DataLoader(test_dataset, batch_size=args.test_bsz, shuffle=True)

    world_size = len(str(args.learners).split('-')) + 1
    this_rank = args.this_rank

    init_myprocesses(this_rank, world_size, model, test_data,
                                                  q, param_q, stop_signal, run, args.backend)
    #p = Process(target=init_myprocesses, args=(this_rank, world_size, model,test_data,
    #                                               q, param_q, stop_signal, run, "gloo"))
    #p.start()
    #p.join()
    manager.shutdown()
