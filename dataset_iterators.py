

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

# For setting a time limit on a process
# For some meta-grammars, you can get stuck in non-terminating
# recursion (or in recursion that will eventually terminate, but only
# after recursing much more than we want). This time limit allows us
# to cut a process short if it is taking too long, to avoid such scenarios.
# Code from here: https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def expand_features(features, max_length=None):
    new_features = []

    for feature in features:
        if feature == 0:
            new_features.append(1)
            new_features.append(0)
        else:
            new_features.append(0)
            new_features.append(1)

    if max_length is not None:
        for _ in range(max_length - len(features)):
            new_features.append(0)
            new_features.append(0)

    return new_features

def generate_all_binary_features_of_max_size(size):
    features = {}

    features[1] = [[0], [1]]

    for i in range(2, size+1):
        features[i] = []
        for elt in features[i-1]:
            features[i].append([0] + elt)
            features[i].append([1] + elt)

    return features


def dnf_dataset(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=True):

    possible_features = generate_all_binary_features_of_max_size(max_n_features)

    def create_dnf_dataset(seed):
        random.seed(seed)
        np.random.seed(seed)

        n_features = random.choice(list(range(min_n_features, max_n_features+1)))
        feature_values = possible_features[n_features][:]

        if reject_sampling:
            all_tf = True
            while all_tf:
                hyp = DNFHypothesis(n_features, no_true_false_top, b)
                hyp_labels = []
                hyp_labels_with_outliers = []
                for features in feature_values:
                    hyp_labels.append(hyp.function(features))
                    hyp_labels_with_outliers.append(hyp.function_with_outliers(features))
                    # print(features, hyp.function(features), hyp.function_with_outliers(features))
                for i in range(len(hyp_labels)-1):
                    if hyp_labels[i] != hyp_labels[i+1]:
                        all_tf = False
                        break
                if all_tf:
                    for i in range(len(hyp_labels_with_outliers)-1):
                        if hyp_labels_with_outliers[i] != hyp_labels_with_outliers[i+1]:
                            all_tf = False
                            break
        else:
            hyp = DNFHypothesis(n_features, no_true_false_top, b)

        # The training set can have repeats and includes outliers
        n_train = random.choice(list(range(min_n_train, max_n_train+1)))
        train_inputs = [random.choice(feature_values) for _ in range(n_train)]
        train_labels = []
        for train_input in train_inputs:
            # Generate labels allowing for outliers
            train_label = hyp.function_with_outliers(train_input)
            train_labels.append(train_label)

        # The test set has no repeats; just one copy of every possible
        # set of feature values.
        # It also doesn't include outliers - hence 'hyp.function' instead
        # of 'hyp.function_with_outliers'
        test_inputs = feature_values
        test_labels = [hyp.function(feature_value) for feature_value in feature_values]


        # Expand input featuress into one-hot encodings
        train_inputs = [expand_features(x, max_length=max_n_features) for x in train_inputs]
        test_inputs = [expand_features(x, max_length=max_n_features) for x in test_inputs]

        if train_batch_size is None:
            batch_size = len(train_inputs)
        else:
            batch_size = train_batch_size

        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.FloatTensor(train_labels).unsqueeze(1), 
        "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.FloatTensor(test_labels).unsqueeze(1), 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_inputs), "rule" : hyp.name}

        return batch

    return create_dnf_dataset

def random_dataset(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False):

    possible_features = generate_all_binary_features_of_max_size(max_n_features)

    def create_random_dataset(seed):
        random.seed(seed)
        np.random.seed(seed)

        n_features = random.choice(list(range(min_n_features, max_n_features+1)))
        feature_values = possible_features[n_features][:]

        # The training set is a subset of all 16 possible inputs
        n_train = random.choice(list(range(min_n_train, max_n_train+1)))
        train_inputs = random.sample(feature_values, n_train)
        train_labels = np.random.randint(2, size=n_train)

        # The test set is identical to training
        test_inputs = train_inputs
        test_labels = train_labels

        # Expand input featuress into one-hot encodings
        train_inputs = [expand_features(x, max_length=max_n_features) for x in train_inputs]
        test_inputs = [expand_features(x, max_length=max_n_features) for x in test_inputs]

        if train_batch_size is None:
            batch_size = len(train_inputs)
        else:
            batch_size = train_batch_size

        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.FloatTensor(train_labels).unsqueeze(1), 
        "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.FloatTensor(test_labels).unsqueeze(1), 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_inputs)}

        return batch

    return create_random_dataset
    
def marble_dataset(min_n_features=1, max_n_features=1, min_n_train=9, max_n_train=9, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False, alpha = 10, beta_0 = 0.2, beta_1 = 0.8):
    
    def create_marble_dataset(seed):
        random.seed(seed)
        np.random.seed(seed)

        n_train = random.choice(list(range(min_n_train, max_n_train+1)))
        
        # "1" being token that signifies marble
        train_inputs = [[1] for i in range(n_train)]
        test_inputs = [[1] for i in range(n_train)]

        # denotes marbles being black/white 
        labels = sample_marbles(seed, n_sets=2, n_marbles=n_train, alpha = alpha, beta_0 = beta_0, beta_1=beta_1)

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

    return create_marble_dataset

# dataset to test whether meta-meta-learning is sufficient
def marble_dataset_2(max_n_train = 27, n_partitions = 3):
    
    def create_marble_dataset_2(seed):
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
    
    return create_marble_dataset_2

# dataset to test whether meta-meta-learning is sufficient, variable length
def marble_dataset_3(min_marbles = 10 , max_marbles = 20, min_bags = 4, max_bags = 8):

    def create_marble_dataset_3(seed):
    
        random.seed(seed)
        np.random.seed(seed)

        lambda_param = 1
        n_colors = 2
        alpha = 0.0
        while alpha == 0: 
            alpha = torch.distributions.Exponential(lambda_param).sample()

        beta = torch.distributions.Dirichlet(torch.ones(n_colors)).sample()

        n_train_bags = random.choice(list(range(min_bags, max_bags+1)))
        n_test_bags = random.choice(list(range(min_bags, max_bags+1)))

        n_marbles_per_bag =  random.choice(list(range(min_marbles, max_marbles+1)))

        train_labels = sample_marbles_transformers(seed, alpha, beta, n_train_bags, n_marbles_per_bag).view(1, -1).squeeze()
        test_labels = sample_marbles_transformers(seed, alpha, beta, n_test_bags, n_marbles_per_bag).view(1, -1).squeeze()

        train_inputs = [[j] for i in range(n_train_bags) for j in [i]*n_marbles_per_bag]
        test_inputs = [[j] for i in range(n_test_bags) for j in [i]*n_marbles_per_bag]

        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels.long()).unsqueeze(1), 
        "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels.long()).unsqueeze(1), 
        "train_batch_size" : 1, "eval_batch_size" : len(test_inputs)}
        
        return batch
    
    return create_marble_dataset_3

# meta-meta-learning using transformers (which is just meta-learning in transformers)
def meta_transformer_dataset(min_marbles = 10 , max_marbles = 20, min_bags = 4, max_bags = 8):
    
    def create_meta_transformer_dataset(seed):

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

        n_marbles_per_bag =  random.choice(list(range(min_marbles, max_marbles+1)))

        train_labels = sample_marbles_transformers(seed, alpha, beta, n_train_bags, n_marbles_per_bag)
        test_labels = sample_marbles_transformers(seed, alpha, beta, n_test_bags, n_marbles_per_bag)

        train_inputs = torch.cat((torch.full((n_train_bags, 1), 9), train_labels), dim=1)[:, :-1]
        test_inputs = torch.cat((torch.full((n_test_bags, 1), 9), test_labels), dim=1)[:, :-1]

        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels.long()).unsqueeze(1), 
        "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels.long()).unsqueeze(1), 
        "train_batch_size" : 1, "eval_batch_size" : len(test_inputs)}

        return batch
    
    return create_meta_transformer_dataset

# meta-meta-learning using transformers (which is just meta-learning in transformers)
def meta_transformer_diff_len_bags_dataset(min_marbles = 10 , max_marbles = 20, min_bags = 4, max_bags = 8):
    
    def create_meta_transformer_diff_len_bags_dataset(seed):

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
    
    return create_meta_transformer_diff_len_bags_dataset


# dataset to test whether meta-meta-learning is sufficient
# THIS IMPLEMENTATION IS WRONG
# def marble_dataset_3(n_partitions = 20, min_partition_size = 5, max_partition_size = 20):
    
#     def create_marble_dataset_3(seed):
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         lambda_param = 1
#         n_colors = 2

#         train_inputs = []
#         test_inputs = []
#         train_labels = []
#         test_labels = []

#         #loop over total number of partitions
#         for p in range(n_partitions):

#             partition_size =  random.choice(list(range(min_partition_size, max_partition_size+1)))

#             train_inputs.extend([[p] for i in [p]*partition_size])
#             train_labels.extend(sample_marbles_2(seed, n_sets = 1, n_marbles = partition_size, n_theta=1)[0])

#         for p in range(n_partitions):
#             partition_size =  random.choice(list(range(min_partition_size, max_partition_size+1)))
#             test_inputs.extend([[p] for i in [p]*partition_size])
#             test_labels.extend(sample_marbles_2(seed, n_sets = 1, n_marbles = partition_size, n_theta=1)[0])

#         batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels).unsqueeze(1), 
#         "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels).unsqueeze(1), 
#         "train_batch_size" : 1, "eval_batch_size" : len(test_inputs)}

#         return batch
    
#     return create_marble_dataset_3

# dataset for meta-meta-learning 
# def meta_meta_marble_dataset(min_marbles = 5, max_marbles = 20, min_bags = 3, max_bags = 6):
    
#     def create_meta_meta_marble_dataset(seed):
#         random.seed(seed)
#         np.random.seed(seed)

#         n_train_marbles = random.choice(list(range(min_marbles, max_marbles+1)))
#         n_test_marbles = random.choice(list(range(min_marbles, max_marbles+1)))
#         n_train_bags = random.choice(list(range(min_bags, max_bags+1)))
#         n_test_bags = random.choice(list(range(min_bags, max_bags+1)))

#         # not [[1], [1], [1], [2], [2], [2]] but [[1], [1], [1]], [[2], [2], [2]]
#         train_inputs = [[j] for i in range(n_partitions) for j in [i]*n_marbles_per_bag]
        
#         test_inputs = [[j] for j in [n_partitions]*n_marbles_per_bag]

#         # denotes marbles being black/white 
#         labels = sample_marbles_2(seed, n_sets=2, n_marbles=n_train, n_theta=n_partitions)

#         train_labels = labels[0]
#         test_labels = labels[1]

#         batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels).unsqueeze(1), 
#         "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels).unsqueeze(1), 
#         "train_batch_size" : 1, "eval_batch_size" : len(test_inputs)}

#         return batch
    
#     return create_meta_meta_marble_dataset

if __name__ == "__main__":
    # print("DNF DATASET")
    # create_dnf_dataset = dnf_dataset(4)
    # for i in range(1):
    #     print(create_dnf_dataset(i))
    #     print("")
    
    # print("RANDOM DATASET")
    # create_random_dataset = random_dataset(4)
    # for i in range(2):
    #     print(create_random_dataset(i))
    #     print("")
    marble_dataset_2(1)

