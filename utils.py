
import math
import torch
import logging

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def meta_mini_batches_from_batch(batch, train_batch_size, test_batch_size):
    train_mini_batches = []
    test_mini_batches = []

    n_train_mini_batches = math.ceil(len(batch["input_ids"])*1.0 / train_batch_size)

    for train_mini_batch_index in range(n_train_mini_batches):
        train_start_index = train_mini_batch_index*train_batch_size
        train_end_index = (train_mini_batch_index+1)*train_batch_size

        train_mini_batch = {"input_ids" : batch["input_ids"][train_start_index:train_end_index], "labels" : batch["labels"][train_start_index:train_end_index]}
        train_mini_batches.append(train_mini_batch)

    if test_batch_size is not None:
        n_test_mini_batches = math.ceil(len(batch["test_input_ids"])*1.0 / test_batch_size)
        for test_mini_batch_index in range(n_test_mini_batches):
            test_start_index = test_mini_batch_index*test_batch_size
            test_end_index = (test_mini_batch_index+1)*test_batch_size

            test_mini_batch = {"input_ids" : batch["test_input_ids"][test_start_index:test_end_index], "labels" : batch["test_labels"][test_start_index:test_end_index]}
            test_mini_batches.append(test_mini_batch)

    return train_mini_batches, test_mini_batches



