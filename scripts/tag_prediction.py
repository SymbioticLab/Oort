import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import pickle
import h5py as h5
from utils.models import *


def get_stackoverflow_datasets(vocab_tokens_size=10000,
                               vocab_tags_size=500,
                               max_training_elements_per_user=500,
                               client_batch_size=100,
                               client_epochs_per_round=1,
                               num_validation_examples=10000):



    


def train(vocab_tokens_size=10000, vocab_tags_size=500):

    batch_size = 100
    learning_rate = 0.001

    # Logistic regression model
    model = create_logistic_model(vocab_tokens_size, vocab_tags_size)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

    return None