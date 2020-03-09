
import argparse
import math
import os, shutil
import sys
sys.path.append("../")
import time
import logging
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import threading
import random

#import numpy as np
import torch
import torch.distributed as dist
from utils.divide_data import partition_dataset, select_dataset, DataPartitioner
from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import MySGD, test_model
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as tormodels
import argparse
import os, shutil
import random
import sys
import time
import logging
#from clientSampler import ClientSampler
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

print(torch.__version__)

#device = torch.device("cuda")
#print(torch.rand(10).cuda())
