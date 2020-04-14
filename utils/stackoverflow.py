from __future__ import print_function
import warnings
import os
import os.path
import torch
import time
import pickle
import h5py as h5
import torch.nn.functional as F

class stackoverflow():
    """
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = []
    MAX_SEQ_LEN = 20000

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True):
        self.train = train  # training set or test set
        self.root = root
        #self.transform = transform
        #self.target_transform = target_transform
        """
        if self.train:
            self.data_file = self.training_file
        else:
            self.data_file = self.test_file

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You have to download it')
        """

        self.train_file = 'stackoverflow_train.h5'
        self.test_file = 'stackoverflow_test.h5'
        self.train = train


        # load data and targets
        self.data, self.targets, self.dict = self.load_file(self.root)

    def __getitem__(self, index):
        """
        Args:xx
            index (int): Index

        Returns:
            tuple: (text, tags)
        """

        return self.data[index], self.targets[index]

    def __mapping_dict__(self):
        return self.dict

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def create_tag_vocab(self, vocab_size, path):
        """Creates vocab from `vocab_size` most common tags in Stackoverflow."""
        tags_file = "vocab_tags.txt"
        with open(path + tags_file, 'rb') as f:
            tags = pickle.load(f)
        return tags[:vocab_size]


    def create_token_vocab(self, vocab_size, path):
        """Creates vocab from `vocab_size` most common words in Stackoverflow."""
        tokens_file = "vocab_tokens.txt"
        with open(path + tokens_file, 'rb') as f:
            tokens = pickle.load(f)
        return tokens[:vocab_size]

    def load_file(self, path):

        # First, get the token and tag dict

        vocab_tokens_size = 10000
        vocab_tags_size = 500
        vocab_tokens = self.create_token_vocab(vocab_tokens_size, path)
        vocab_tags = self.create_tag_vocab(vocab_tags_size, path)

        vocab_tokens_dict = {k: v for v, k in enumerate(vocab_tokens)}
        vocab_tags_dict = {k: v for v, k in enumerate(vocab_tags)}

        # Load the traning data
        if self.train:
            train_file = h5.File(path + self.train_file, "r")
        else:
            train_file = h5.File(path + self.test_file, "r")
        print(self.train)
        text, target_tags = [], []

        # Mapping from sample id to target tag
        mapping_dict = {}
        count = 0

        client_list = list(train_file['examples'])

        for clientId, client in enumerate(client_list):
            tags_list = list(train_file['examples'][client]['tags'])
            tokens_list = list(train_file['examples'][client]['tokens'])

            title = str(train_file['examples'][client]['title'])
            for tags, tokens in zip(tags_list, tokens_list):
                tokens_list = [s for s in tokens.decode("utf-8").split() if s in vocab_tokens_dict]
                tags_list = [s for s in tags.decode("utf-8").split('|') if s in vocab_tags_dict]
                if not tokens_list or not tags_list:
                    continue
                # Lookup tensor
                tokens = torch.tensor([vocab_tokens_dict[i] for i in tokens_list], dtype=torch.long)
                tokens = F.one_hot(tokens, vocab_tokens_size).float()
                tokens = tokens.mean(0)

                tags = torch.tensor([vocab_tags_dict[i] for i in tags_list], dtype=torch.long)
                tags = F.one_hot(tags, vocab_tags_size).float()
                tags = tags.sum(0)

                count += 1
                mapping_dict[count] = clientId
                text.append(tokens)
                target_tags.append(tags)

        return text, target_tags, mapping_dict
