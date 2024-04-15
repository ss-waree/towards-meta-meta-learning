
import os
import sys

import math
import random
import numpy as np
from collections import Counter
import logging 

import torch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast

from utils import *

  
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')




################################################################################################
# Abstract base classes for datasets and data splits
# For your specific use case, you should create new classes that inherit from these
# and override the attributes and methods listed in the abstract class
################################################################################################

# Abstract base class for a dataset
class Dataset:

    def __init__(self):
        
        # Should include 3 data splits - train, valid, and test
        self.train = None
        self.valid = None
        self.test = None

    def prepare_input(self, batch):
        
        # Keys in the batch that can be converted into tensors
        tensorizable_keys = ["input_ids", "labels", "attention_mask", "test_input_ids", "test_labels", "test_attention_mask"]

        prepared_batch = {}

        # Convert to tensors and put on the proper device (CPU or GPU)
        for key in batch:
            if key in tensorizable_keys:
                prepared_batch[key] = batch[key].to(device) #torch.LongTensor(batch[key]).to(device)
            else:
                prepared_batch[key] = batch[key]

        # If it's a meta-batch, deal with labels for padding in the test set
        if "test_input_ids" in prepared_batch:

            train_batches, test_batches = meta_mini_batches_from_batch(prepared_batch, prepared_batch["train_batch_size"], prepared_batch["eval_batch_size"])

            prepared_batch["train_batches"] = train_batches
            prepared_batch["test_batches"] = test_batches


        return prepared_batch



# Abstract base class for a data split
class DataSplit:

    def __init__(self):

        # There are no specific attributes that need to be included
        pass

    def __iter__(self):
        return self

    def __len__(self):
        # Should return the length of the data split (i.e., the number of batches in it)
        raise NotImplementedError

    def __next__(self):
        # Should return the next batch (or raise StopIteration if you've
        # reached the end of the data split)
        raise NotImplementedError

    def reset(self):
        # Should reset the data split so that it can be iterated over again
        raise NotImplementedError



################################################################################################
# Data classes for meta-language modeling: Meta-learning where the task is language
# modeling (next-word prediction) and each episode is a different language
# Here, the data are not packed: each input to the model is exactly one sentence
################################################################################################


# A dataset of datasets
# That is, each dataset is one corpus - one episode for meta-training
class MetaLogicDataset(Dataset):
    # create_dataset: Function that takes in a random seed and returns a dataset.
    #     The dataset that is returned should be a dict with the following key, value pairs:
    #     - ["train", a list of training strings, with space-delimited tokens within each string]
    #     - ["test", a list of test strings, with space-delimited tokens within each string]
    # meta_train_size, meta_valid_size, meta_test_size: Number of tasks
    #     to include in each meta data split. You can leave meta_train_size
    #     as None to have it generate indefinitely.
    # integer_vocab_size: If all the words in the vocab are positive integers,
    #     gives the max integer in the vocab.
    def __init__(self, create_dataset=None, meta_train_size=None, meta_valid_size=None, meta_test_size=None):

        super(MetaLogicDataset, self).__init__()

        self.create_dataset = create_dataset

        self.meta_train_size = meta_train_size
        self.meta_valid_size = meta_valid_size
        self.meta_test_size = meta_test_size

        # Define each dataset split
        self.train = MetaLogicDataSplit(length=self.meta_train_size, create_dataset=self.create_dataset, initial_index=meta_valid_size+meta_test_size, prepare_input=self.prepare_input)
        self.valid = MetaLogicDataSplit(length=self.meta_valid_size, create_dataset=self.create_dataset, initial_index=0, prepare_input=self.prepare_input)
        self.test = MetaLogicDataSplit(length=self.meta_test_size, create_dataset=self.create_dataset, initial_index=self.meta_valid_size, prepare_input=self.prepare_input)



# Based on a dataset iterator, rather than a dataset file
# The meta training set, meta validation set, and meta test set are
# all MetaDataSplits
# - length: Number of episodes
# - tokenize: Function for tokenizing a dataset (takes in dataset, returns tokenized dataset)
# - initial_index: index of the first episode (so, it runs from initial_index to initial_index + length)
# - context_size: context size of the model, so we can truncate text to that
# - remember_languages: When withholding certain languages, remember all languages we have
#   seen so that we know if they should be withheld or not
class MetaLogicDataSplit(DataSplit):

    def __init__(self, length=None, create_dataset=None, initial_index=None, context_size=None, prepare_input=None):

        super(MetaLogicDataSplit, self).__init__()

        self.length = length
        self.create_dataset = create_dataset

        self.initial_index = initial_index
        self.current_index = initial_index

        self.prepare_input = prepare_input

    def __len__(self):
        return self.length

    def __next__(self):

        if self.current_index == self.initial_index + self.length:
            # We've used the whole dataset
            raise StopIteration

        to_return = self.create_dataset(self.current_index)
        to_return = self.prepare_input(to_return)

        self.current_index += 1
        return to_return

    def reset(self, offset=None):
        self.current_index = self.initial_index



if __name__ == "__main__":
    # You can use this space to try running things in this file
    pass