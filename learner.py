# -*- coding: utf-8 -*-
# Thanks to qihua Zhou

from core.argParser import args
import os, shutil, sys, time, datetime, logging, pickle
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import random, math, gc, copy
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
from utils.nlp import *
from utils.inception import *
from utils.stackoverflow import *

#device = torch.device(args.to_device)
#torch.set_num_threads(int(args.threads))

logDir = os.getcwd() + "/../../models/" + args.model + '/' + args.time_stamp + '/learner/'
logFile = logDir + 'log_'+str(args.this_rank)
modelDir = os.getcwd() + "/../../models/"  + args.model
modelPath = modelDir+'/'+str(args.model)+'.pth.tar' if args.model_path is None else args.model_path

def init_logging():
    global logDir

    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    # files = [logFile, '/tmp/sampleDistribution']
    # for file in files:
    #     with open(file, "w") as fout:
    #         pass

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
deviceId = None
device = torch.device(args.to_device)

for i in range(4):
    try:
        device = torch.device(cudaPrefix+str(i))
        torch.cuda.set_device(i)
        logging.info(torch.rand(1).to(device=device))
        deviceId = i
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
last_model_tensors = []
nextClientIds = None
global_data_iter = {}
global_client_profile = {}
global_optimizers = {}

workers = [int(v) for v in str(args.learners).split('-')]

# initiate for nlp
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2') if args.task =='nlp' else None

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
    global tokenizer, device

    outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47, 'openImg': 596}

    if args.data_set == 'Mnist':
        train_transform, test_transform = get_data_transform('mnist')

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                       transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                                      transform=test_transform)

    elif args.data_set == 'cifar10':
        train_transform, test_transform = get_data_transform('cifar')
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                        transform=test_transform)

    elif args.data_set == "imagenet":
        train_transform, test_transform = get_data_transform('imagenet')
        train_dataset = datasets.ImageNet(args.data_dir, split='train', download=False, transform=train_transform)
        test_dataset = datasets.ImageNet(args.data_dir, split='val', download=False, transform=test_transform)

    elif args.data_set == 'emnist':
        test_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=False, download=True, transform=transforms.ToTensor())
        train_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=True, download=True, transform=transforms.ToTensor())

    elif args.data_set == 'openImg':
        transformer_ns = 'openImg' if args.model != 'inception_v3' else 'openImgInception'
        train_transform, test_transform = get_data_transform(transformer_ns) 
        train_dataset = OPENIMG(args.data_dir, train=True, transform=train_transform)
        test_dataset = OPENIMG(args.data_dir, train=False, transform=test_transform)

    elif args.data_set == 'blog':
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False) 
        test_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    elif args.data_set == 'stackoverflow':
        train_dataset = stackoverflow(args.data_dir, train=True)
        test_dataset = stackoverflow(args.data_dir, train=False)
    else:
        print('DataSet must be {}!'.format(['Mnist', 'Cifar', 'openImg', 'blog']))
        sys.exit(-1)

    logging.info("====Initialize the model")

    if args.task == 'nlp':
        # we should train from scratch
        config = AutoConfig.from_pretrained(os.path.join(args.data_dir, 'albert-base-v2-config.json'))
        model = AutoModelWithLMHead.from_config(config)
    elif args.task == 'tag-one-sample':
        # Load LR model for tag prediction
        model = LogisticRegression(args.vocab_token_size, args.vocab_tag_size)
    else:
        if args.model == 'mnasnet':
            model = MnasNet(num_classes=outputClass[args.data_set])
        elif args.model == "lr":
            model = LogisticRegression(args.input_dim, outputClass[args.data_set])
        else:
            model = tormodels.__dict__[args.model](num_classes=outputClass[args.data_set])

    logging.info("====Initialize client configuration")

    # initiate the device information - normalized computation speed by enforcing sleeping, bandwidth
    client_cfg = {}

    return model, train_dataset, test_dataset, client_cfg

# ================== Scorer =================== #
def run_forward_pass(model, test_data):
    test_loss = 0.
    test_len = 0.
    totalLoss = None
    # we want avg{Loss^2}

    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device) if args.task != 'tag' else torch.nn.BCEWithLogitsLoss(reduction='none').to(device=device)
   
    gradientSamples = []

    for data, target in test_data:
        data, target = Variable(data).to(device=device), Variable(target).to(device=device)
 
        if args.model != 'inception_v3':
            output = model(data)

            if args.model == 'googlenet':
                loss1 = criterion(output, target)
                loss2 = criterion(aux1, target)
                loss3 = criterion(aux2, target)
                loss = loss1 + 0.3 * (loss2 + loss3)
            else:
                loss = criterion(output, target)
        else:
            output, aux_outputs = model(data)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_outputs, target)
            loss = loss1 + 0.3*loss2

        #loss = criterion(output, target)

        for l in loss.tolist():
            test_loss += l**2
        # loss = criterion(output, target)
        # test_loss += loss.data.item()

        test_len += len(target)

    # loss function averages over batch size
    #test_loss /= float(len(test_data))

    return (test_loss/float(test_len))

def run_backward_pass(model, test_data):
    test_loss = 0.
    test_len = 0.
    totalLoss = None
    gradient_norm = 0

    #model.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device=device) if args.task != 'tag' else torch.nn.BCEWithLogitsLoss().to(device=device)
    optimizer = MySGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    gradientSamples = []

    for data, target in test_data:
        data, target = Variable(data).to(device=device), Variable(target).to(device=device)
        #output = model(data)

        if args.model != 'inception_v3':
            output = model(data)

            if args.model == 'googlenet':
                loss1 = criterion(output, target)
                loss2 = criterion(aux1, target)
                loss3 = criterion(aux2, target)
                loss = loss1 + 0.3 * (loss2 + loss3)
            else:
                loss = criterion(output, target)
        else:
            output, aux_outputs = model(data)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_outputs, target)
            loss = loss1 + 0.4*loss2

        test_loss += loss.data.item()
        test_len += len(target)

        optimizer.zero_grad()
        loss.backward()

        # sum the gradient norm of samples
        for p in list(filter(lambda p: p.grad is not None, model.parameters())):
            gradient_norm += p.grad.data.norm(2).item()

    # loss function averages over batch size
    gradient_norm /= float(len(test_data))

    return gradient_norm

def collate(examples: List[torch.Tensor]):
    global tokenizer

    if tokenizer._pad_token is None:
        return (pad_sequence(examples, batch_first=True), None)
    return (pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id), None)

def run_client(clientId, cmodel, iters, learning_rate, argdicts = {}):
    global global_trainDB, global_data_iter, last_model_tensors, tokenizer

    curBatch = -1

    if args.task != 'nlp':
        optimizer = MySGD(cmodel.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in cmodel.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in cmodel.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=args.adam_epsilon)
        #optimizer = torch.optim.SGD(cmodel.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device) if args.task != 'tag' else torch.nn.BCEWithLogitsLoss(reduction='none').to(device=device)
   
    train_data_itr_list = []

    if clientId not in global_data_iter:
        client_train_data = select_dataset(
                                clientId, global_trainDB, 
                                batch_size=args.batch_size, 
                                collate_fn=collate if args.task =='nlp' else None
                            )

        train_data_itr = iter(client_train_data)
        total_batch_size = len(train_data_itr)
        global_data_iter[clientId] = [train_data_itr, curBatch, total_batch_size, argdicts['iters']]
    else:
        [train_data_itr, curBatch, total_batch_size, epo] = global_data_iter[clientId]

    local_trained = 0
    numOfPreWarmUp = 1
    epoch_train_loss = None
    comp_duration = 0.
    norm_gradient = 0.
    count = 0

    train_data_itr_list.append(train_data_itr)
    run_start = time.time()

    numOfFailures = 0
    numOfTries = 5

    cmodel.train()
    # TODO: if indeed enforce FedAvg, we will run fixed number of epochs, instead of iterations

    for itr in range(iters):
        it_start = time.time()

        fetchSuccess = False
        while not fetchSuccess and numOfFailures < numOfTries:
            try:
                try:
                    if args.task == 'nlp':
                        # target is None in this case
                        (data, _) = next(train_data_itr_list[0])
                        data, target = mask_tokens(data, tokenizer, args) if args.mlm else (data, data)
                    else:
                        (data, target) = next(train_data_itr_list[0])

                    fetchSuccess = True
                except Exception:
                    try:
                        train_data_itr_list[0]._shutdown_workers()
                        del train_data_itr_list[0]
                    except Exception as e:
                        logging.info("====Error {}".format(e))

                    # reload data after finishing the epoch
                    numToWarmUp = numOfPreWarmUp - len(train_data_itr_list)

                    for i in range(numToWarmUp):
                        tempData = select_dataset(
                            clientId, global_trainDB, 
                            batch_size=args.batch_size, 
                            collate_fn=collate if args.task =='nlp' else None
                        )
                        train_data_itr_list.append(iter(tempData))

                    if args.task == 'nlp':
                        (data, _) = next(train_data_itr_list[0])
                        data, target = mask_tokens(data, tokenizer, args) if args.mlm else (data, data)
                    else:
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

        if args.task != 'nlp':
            optimizer.zero_grad()
        comp_start = time.time()

        if args.task == 'nlp':
            cmodel.train()
            outputs = cmodel(data, masked_lm_labels=target) if args.mlm else cmodel(data, labels=target)
            loss = outputs[0]
            #torch.nn.utils.clip_grad_norm_(cmodel.parameters(), args.max_grad_norm)
        else:
            if args.model != 'inception_v3':
                if args.model == 'googlenet':
                    output, aux1, aux2 = cmodel(data)
                    loss1 = criterion(output, target)
                    loss2 = criterion(aux1, target)
                    loss3 = criterion(aux2, target)
                    loss = loss1 + 0.3 * (loss2 + loss3)
                else:
                    output = cmodel(data)
                    loss = criterion(output, target)
            else:
                output, aux_outputs = cmodel(data)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_outputs, target)
                loss = loss1 + 0.4*loss2

        temp_loss = 0.

        if args.task != 'nlp':
            if args.task == 'tag':
                if itr >= (iters - total_batch_size - 1):
                    for l in loss.tolist():
                        for i in l:
                            temp_loss += i**2
            else:
                if itr >= (iters - total_batch_size - 1):
                    for l in loss.tolist():
                        temp_loss += l**2
        else:
            temp_loss = loss.item()

        temp_loss = temp_loss/float(len(target))

        # only measure the loss of the first epoch
        if itr < total_batch_size:
            if epoch_train_loss is None:
                epoch_train_loss = temp_loss
            else:
                epoch_train_loss = (1. - args.loss_decay) * epoch_train_loss + args.loss_decay * temp_loss

        count += len(target)

        # ========= Define the backward loss ==============
        if args.task != 'nlp':
            loss = torch.mean(loss)
            loss.backward()

            delta_w = optimizer.get_delta_w(learning_rate)

            if not args.proxy_avg:
                for idx, param in enumerate(cmodel.parameters()):
                    param.data -= delta_w[idx].to(device=device)
            else:
                for idx, param in enumerate(cmodel.parameters()):
                    param.data -= delta_w[idx].to(device=device)
                    param.data += learning_rate * args.proxy_mu * (last_model_tensors[idx] - param.data)
        else:
            loss.backward()
            optimizer.step()
            cmodel.zero_grad()

        comp_duration = (time.time() - comp_start)
    
        logging.info('For client {}, upload iter {}, epoch {}, Batch {}/{}, Loss:{} | TotalTime {} | Comptime: {} \n'
                    .format(clientId, argdicts['iters'], int(curBatch/total_batch_size),
                    (curBatch % total_batch_size), total_batch_size, round(loss.data.item(), 4), 
                    round(time.time() - it_start, 4), round(comp_duration, 4)))

    # remove the one with LRU
    if len(global_client_profile) > args.max_iter_store:
        allClients = global_data_iter.keys()
        rmClient = sorted(allClients, key=lambda k:global_data_iter[k][3])[0]

        del global_data_iter[rmClient]

    # save the state of this client if # of batches > iters, since we want to pass over all samples at least one time
    if total_batch_size > iters and len(train_data_itr_list) > 0:
        global_data_iter[clientId] = [train_data_itr_list[0], curBatch, total_batch_size, argdicts['iters']]
    else:
        for loader in train_data_itr_list:
            try:
                loader._shutdown_workers()
            except Exception as e:
                pass
        del train_data_itr_list
        del global_data_iter[clientId]

    # we only transfer the delta_weight
    model_param = [(param.data - last_model_tensors[idx]).cpu().numpy() for idx, param in enumerate(cmodel.parameters())]
    
    time_spent = time.time() - run_start

    # add bias to the virtual clock, computation x (# of trained samples) + communication
    if clientId in global_client_profile:
        time_cost = global_client_profile[clientId][0] * count + global_client_profile[clientId][1]
    else:
        time_cost = time_spent

    speed = 0
    isSuccess = True
    if count > 0:
        speed = time_spent/float(count) 
        #epoch_train_loss /= float(count)
    else:
        isSuccess = False
        logging.info("====Failed to run client {}".format(clientId))

    return model_param, epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess

def run(rank, model, train_data, test_data, queue, param_q, stop_flag, client_cfg):
    logging.info("====Worker: Start running")

    global nextClientIds, global_trainDB, last_model_tensors
    criterion = torch.nn.CrossEntropyLoss().to(device=device) if args.task != 'tag' else torch.nn.BCEWithLogitsLoss().to(device=device)
   
    global_trainDB = train_data
    startTime = time.time()

    # Fetch the initial parameters from the server side (we called it parameter_server)
    last_model_tensors = []
    #model.train()
    model = model.to(device=device)

    for idx, param in enumerate(model.parameters()):
        dist.broadcast(tensor=param.data, src=0)
    
    for idx, param in enumerate(model.parameters()): 
        last_model_tensors.append(copy.deepcopy(param.data))

    print('Begin!')
    logging.info('\n' + repr(args) + '\n')

    learning_rate = args.learning_rate

    testResults = [0, 0, 0, 0]
    # first run a forward pass
    test_loss, acc, acc_5, testResults = test_model(rank, model, test_data, criterion=criterion, tokenizer=tokenizer)
    uploadEpoch = 0

    last_test = time.time()

    tempModelPath = logDir+'/model_'+str(args.this_rank)+'.pth.tar'

    for epoch in range(1, int(args.epochs) + 1):
        try:
            if epoch % args.decay_epoch == 0:
                learning_rate = max(args.min_learning_rate, learning_rate * args.decay_factor)

            trainedModels = []
            preTrainedLoss = []
            trainedSize = []
            trainSpeed = []
            virtualClock = []
            ranClients = []

            computeStart = time.time()

            if not args.test_only or epoch == 1:
                # dump a copy of model
                with open(tempModelPath, 'wb') as fout:
                    pickle.dump(model, fout)

                for nextClientId in nextClientIds:
                    # roll back to the global model for simulation
                    with open(tempModelPath, 'rb') as fin:
                        model = pickle.load(fin)
                        model = model.to(device=device)

                    if args.score_mode == 'norm':
                        # need to get the individual norm of samples
                        backward_dataset = select_dataset(nextClientId, global_trainDB, batch_size=1, isTest=True)
                        gradient_norm = run_backward_pass(model, backward_dataset)
                        score = gradient_norm

                    _model_param, _loss, _trained_size, _speed, _time, _isSuccess = run_client(
                                clientId=nextClientId, 
                                cmodel=model, 
                                learning_rate=learning_rate, 
                                iters=args.upload_epoch,
                                argdicts={'iters': epoch}
                            )

                    if _isSuccess is False:
                        continue

                    score = -1
                    if args.forward_pass:
                        forward_dataset = select_dataset(nextClientId, global_trainDB, batch_size=args.test_bsz, isTest=True)
                        forward_loss = run_forward_pass(model, forward_dataset)
                        score = forward_loss

                    trainedModels.append(_model_param)
                    preTrainedLoss.append(_loss if score == -1 else score)
                    trainedSize.append(_trained_size)
                    trainSpeed.append(_speed)
                    virtualClock.append(_time)
                    ranClients.append(nextClientId)

                    #gc.collect()
            else:
                logging.info('====Start test round {}'.format(epoch))
                testResults = [0., 0., 0., 0.]
                # designed for testing only
                for nextClientId in nextClientIds:
                    client_dataset = select_dataset(nextClientId, global_trainDB, batch_size=args.test_bsz, isTest=True)
                    test_loss, acc, acc_5, temp_testResults = test_model(rank, model, client_dataset, criterion=criterion, 
                                                                            tokenizer=tokenizer)
                    # merge test results
                    for idx, item in enumerate(temp_testResults):
                        testResults[idx] += item

                uploadEpoch = epoch

            computeEnd = time.time() - computeStart

            # upload the weight
            sendStart = time.time()
            testResults.append(uploadEpoch)
            queue.put({rank: [trainedModels, preTrainedLoss, trainedSize, False, ranClients, trainSpeed, testResults, virtualClock]})
            uploadEpoch = -1
            sendDur = time.time() - sendStart

            logging.info("====Pushing takes {} s".format(sendDur))
            # wait for new models
            receStart = time.time()

            last_model_tensors = []
            for idx, param in enumerate(model.parameters()):
                tmp_tensor = torch.zeros_like(param.data)
                dist.broadcast(tensor=tmp_tensor, src=0)
                param.data = tmp_tensor
                last_model_tensors.append(copy.deepcopy(tmp_tensor))

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

            #logging.info("====Finish receiving ps")
            evalStart = time.time()
            # test the model if necessary
            if epoch % int(args.eval_interval) == 0:
                model = model.to(device=device)
                # forward pass of the training data
                if args.test_train_data:
                    rank_train_data = select_dataset(
                                        this_rank, global_trainDB, batch_size=args.test_bsz, is_rank=rank,
                                        collate_fn=collate if args.task=='nlp' else None
                                      )
                    test_loss, acc, acc_5, testResults = test_model(rank, model, rank_train_data, criterion=criterion, tokenizer=tokenizer)
                else:
                    test_loss, acc, acc_5, testResults = test_model(rank, model, test_data, criterion=criterion, tokenizer=tokenizer)
    
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
    BaseManager.register('get_queue')
    BaseManager.register('get_param')
    BaseManager.register('get_stop_signal')
    manager = BaseManager(address=(args.ps_ip, args.manager_port), authkey=b'queue')
    manager.connect()

    q = manager.get_queue()  # queue for parameter_server signal process
    param_q = manager.get_param()  # init
    stop_signal = manager.get_stop_signal()  # stop

    logging.info("====Start to initialize dataset")

    gc.disable()
    model, train_dataset, test_dataset, client_cfg = init_dataset()
    gc.enable()

    splitTrainRatio = []

    # Initialize PS - client communication channel
    world_size = len(workers) + 1
    this_rank = args.this_rank
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
    test_data = select_dataset(this_rank, testsetPartitioner, batch_size=args.test_bsz, isTest=True, collate_fn=collate if args.task=='nlp' else None)

    stop_flag = Value(c_bool, False)
    init_myprocesses(this_rank, world_size, model, entire_train_data, test_data,
                                          q, param_q, stop_flag,
                                          run, args.backend, client_cfg)
