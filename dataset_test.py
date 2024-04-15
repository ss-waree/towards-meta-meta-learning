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

# dataset for meta-meta-learning is sufficient
def meta_meta_marble_dataset(min_marbles = 5, max_marbles = 20, min_bags = 4, max_bags = 7):
    
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    train_inputs = []
    test_inputs = []

    train_partition_sizes = []
    test_partition_sizes = []

    n_train_bags = random.choice(list(range(min_bags, max_bags+1)))
    n_test_bags = random.choice(list(range(min_bags, max_bags+1)))

    for p in range(n_train_bags):

        partition_size = random.choice(list(range(min_marbles, max_marbles+1)))
        train_partition_sizes.append(partition_size)
        train_inputs.extend([[[p] for i in [p]*partition_size]])

    train_labels = sample_marbles_3(seed, train_partition_sizes)

    for p in range(n_test_bags):

        partition_size =  random.choice(list(range(min_marbles, max_marbles+1)))
        test_partition_sizes.append(partition_size)
        test_inputs.extend([[[p] for i in [p]*partition_size]])

    test_labels = sample_marbles_3(seed, test_partition_sizes)

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels).unsqueeze(1), 
    "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels).unsqueeze(1), 
    "train_batch_size" : 1, "eval_batch_size" : len(test_inputs)}

    return batch


if __name__ == "__main__":
    print(meta_meta_marble_dataset())