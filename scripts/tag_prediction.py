import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import pickle
import h5py as h5
from utils.models import *

def create_logistic_model(vocab_tokens_size, vocab_tags_size):
    """Logistic regression to predict tags of StackOverflow.

    Args:
        vocab_tokens_size: Size of token vocabulary to use.
        vocab_tags_size: Size of token vocabulary to use.

    """
    model = LogisticRegression(vocab_tokens_size, vocab_tags_size)
    return model 

def create_tag_vocab(vocab_size):
    """Creates vocab from `vocab_size` most common tags in Stackoverflow."""
    tags_file = "vocab_tags.txt"
    with open(tags_file, 'rb') as f:
        tags = pickle.load(f)
    return tags[:vocab_size]


def create_token_vocab(vocab_size):
  """Creates vocab from `vocab_size` most common words in Stackoverflow."""
    tokens_file = "vocab_tokens.txt"
    with open(tokens_file, 'rb') as f:
        tokens = pickle.load(f)
    return tokens[:vocab_size]


def get_stackoverflow_datasets(vocab_tokens_size=10000,
                               vocab_tags_size=500,
                               max_training_elements_per_user=500,
                               client_batch_size=100,
                               client_epochs_per_round=1,
                               num_validation_examples=10000):





    vocab_tokens = create_token_vocab(vocab_tokens_size)
    vocab_tags = create_tag_vocab(vocab_tags_size)


    


def train(vocab_tokens_size=10000, vocab_tags_size=500):

    batch_size = 100
    learning_rate = 0.001

    # Logistic regression model
    model = create_logistic_model(vocab_tokens_size, vocab_tags_size)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

    return None