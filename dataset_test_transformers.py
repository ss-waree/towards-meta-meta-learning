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
def meta_transformer_dataset(min_marbles = 10 , max_marbles = 20, min_bags = 4, max_bags = 8):
    
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    lambda_param = 1
    n_colors = 2
    alpha = 0.0
    while alpha == 0: 
        alpha = torch.distributions.Exponential(lambda_param).sample()

    beta = torch.distributions.Dirichlet(torch.ones(n_colors)).sample()

    n_train_bags = random.choice(list(range(min_bags, max_bags+1)))
    n_test_bags = random.choice(list(range(min_bags, max_bags+1)))

    n_marbles_each_bag_train =  np.random.randint(min_marbles, max_marbles + 1, size=n_train_bags)
    n_marbles_each_bag_test =  np.random.randint(min_marbles, max_marbles + 1, size=n_test_bags)

    train_labels = sample_marbles_transformers_2(seed, alpha, beta, n_train_bags, n_marbles_each_bag_train, max_marbles)
    test_labels = sample_marbles_transformers_2(seed, alpha, beta, n_test_bags, n_marbles_each_bag_test, max_marbles)

    train_inputs = torch.cat((torch.full((n_train_bags, 1), 9), train_labels), dim=1)[:, :-1]
    test_inputs = torch.cat((torch.full((n_test_bags, 1), 9), test_labels), dim=1)[:, :-1]

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels.long()).unsqueeze(1), 
    "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels.long()).unsqueeze(1), 
    "train_batch_size" : 1, "eval_batch_size" : len(test_inputs)}

    return batch


if __name__ == "__main__":
    print(meta_transformer_dataset())