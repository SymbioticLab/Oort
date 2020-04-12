import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import pickle
import h5py as h5
import sys
sys.path.append("..")
from utils.stackoverflow import *

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = F.sigmoid(self.linear(x))
        return output




def create_logistic_model(vocab_tokens_size, vocab_tags_size):
    """Logistic regression to predict tags of StackOverflow.

    Args:
        vocab_tokens_size: Size of token vocabulary to use.
        vocab_tags_size: Size of token vocabulary to use.

    """
    model = LogisticRegression(vocab_tokens_size, vocab_tags_size)
    return model




def train(vocab_tokens_size=10000, vocab_tags_size=500):

    batch_size = 100
    learning_rate = 0.001

    # Logistic regression model
    model = create_logistic_model(vocab_tokens_size, vocab_tags_size)


    train_dataset = stackoverflow('/users/xzhu/tag/stackoverflow_tf/', True)
    print(train_dataset.__getitem__(0)[0].size())
    print(train_dataset.__getitem__(0)[1].size())

    print(train_dataset.__getitem__(1)[0].size())
    print(train_dataset.__getitem__(1)[1].size())



train()
