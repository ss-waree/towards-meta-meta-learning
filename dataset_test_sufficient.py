import math
import random
from collections import Counter
import logging

import numpy as np
from utils import *
from dnf_grammar import *
from marble_sampling import *

import signal
from contextlib import contextmanager

import torch

# dataset to test whether meta-meta-learning is sufficient
def marble_dataset_2(max_n_train = 27, n_partitions = 3):

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    n_train = max_n_train

    #partition the n_train into different bags of marbles
    assert n_train % n_partitions == 0, "n_train/partition is not an integer"

    n_marbles_per_bag = int(n_train/n_partitions)
    
    train_inputs = [[j] for i in range(n_partitions) for j in [i]*n_marbles_per_bag]
    # print("train_inputs", train_inputs)
    
    test_inputs = [[j] for i in range(n_partitions) for j in [i]*n_marbles_per_bag]
    # print("test_inputs", test_inputs)

    # denotes marbles being black/white 
    labels,_ = sample_marbles_2(seed, n_sets=2, n_marbles=n_train, n_theta=n_partitions)

    # debugging labels REMOVE FOR EXPERIMENTS
    #labels = np.zeros((2,n_train))

    train_labels = labels[0]
    # print("train", train_labels)
    test_labels = labels[1]
    # print("test", test_labels)

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels).unsqueeze(1), 
    "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels).unsqueeze(1), 
    "train_batch_size" : 1, "eval_batch_size" : len(test_inputs)}

    return batch

if __name__ == "__main__":
    print(marble_dataset_2())