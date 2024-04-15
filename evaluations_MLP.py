
import jsonlines
import logging
import torch
import csv

import copy

from dataset_iterators import *
from dataloading import *
from training import *

from datetime import datetime

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


################################################################################################
# Category evaluations
################################################################################################

def marble(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, num_black=10, num_white=10):
    
    # set up support set
    # set up training batch according to given number of marbles

    n_train = num_black + num_white
    
    # "1" being token that signifies marble
    train_inputs = [[1] for i in range(n_train)]

    # denotes marbles being black/white 
    train_labels = np.zeros(n_train)
    train_labels[num_black:] = 1

    # shuffle with seeds later
    random.shuffle(train_labels)

    # print(train_labels)

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels).unsqueeze(1),  
    "train_batch_size" : 1}

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)
    # training batch here = 1 bag of marbles
    temp_model = simple_train_model(temp_model, training_batch, lr=lr, epochs=epochs, vary_train_batch_size=vary_train_batch_size)

    # evaluate with "marble" token
    batch = {"input_ids" : torch.FloatTensor([[1]])}
    outp = temp_model(batch)["probs"].detach().numpy()

    return outp

def marble_n_runs(model, model_name, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs=10, num_black=10, num_white=10):
    probs_by_index = np.zeros((n_runs, 2))
    
    for i in range(n_runs):
        outputs = marble(model, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, epochs=epochs, num_black = num_black, num_white = num_white)
        probs_by_index[i] = outputs

    n_train = num_black + num_white
    true_theta = np.array([num_black/n_train, num_white/n_train])
    pred_theta = np.sum(probs_by_index, axis = 0)/n_runs
    err = np.abs(true_theta[0]-pred_theta[0]) 

    logging.info("True theta = " + str([true_theta]))
    logging.info("Average predicted theta = " + str(pred_theta))

    # Data to be written to the CSV file
    data = [model_name, num_black, num_white, n_train, true_theta[0], true_theta[1], pred_theta[0], pred_theta[1], err, 1-err] 

    # Open a file in write mode
    with open('prior_results_MLP_diff_len.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def figure3_ii(model, model_name, seed = 0, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1):

    np.random.seed(seed)
    n_partitions = 20
    n_bw_partitions = int(0.5*n_partitions)
    n_marbles_per_bag = 20
    n_marbles = n_partitions * n_marbles_per_bag
    n_sets = 2
    color_final_bag = 0

    # -------------- train for n_partition number of bags ----------------------- #
    
    train_inputs = [[j] for i in range(n_partitions) for j in [i]*n_marbles_per_bag]
    train_inputs.append([n_partitions])
    labels = np.zeros(n_marbles)
    labels_one_partition = np.zeros(n_marbles_per_bag)

    bag_color = np.array([0] * n_bw_partitions + [1] * n_bw_partitions)
    np.random.shuffle(bag_color)

    for t in range(n_partitions):
        if bag_color[t] == 1:
            labels_one_partition = np.ones(n_marbles_per_bag)
        else:
            labels_one_partition = np.zeros(n_marbles_per_bag)
        labels[n_marbles_per_bag*t:n_marbles_per_bag*(t+1)] = labels_one_partition
    
    labels = np.append(labels, color_final_bag)

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(labels).unsqueeze(1),  
    "train_batch_size" : 1}

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)
    
    temp_model, output_arr = simple_train_model_fig3(temp_model, training_batch, lr=lr, epochs=epochs, vary_train_batch_size=vary_train_batch_size)

    # -------------- train on single BLANK marble ----------------------- #
    
    batch = {"input_ids" : torch.FloatTensor([[n_partitions]])}
    output_final = temp_model(batch)["probs"].detach().numpy()

    # -------------- write results -------------------------------------- #
    filename = "figure3_ii_" + model_name + ".csv"

    with open(filename, 'a', newline='') as file:
        for i in range(len(train_inputs)):

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            data = [model_name, current_time, i, train_inputs[i][0], int(labels[i]), output_arr[i][0][0], output_arr[i][0][1]] 
            writer = csv.writer(file) 
            writer.writerow(data)

        data = [model_name, current_time, n_marbles+1, n_partitions, color_final_bag, output_final[0][0], output_final[0][1]] 
        writer = csv.writer(file)
        writer.writerow(data)

    return output_final

def figure3_ii_n_runs(model, model_name, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs = 10):
    probs_by_index = np.zeros((n_runs, 2))
    
    for i in range(n_runs):
        output = figure3_ii(model, model_name, lr=lr, seed = i, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, epochs=epochs)
        probs_by_index[i] = output[0]

    pred_theta = np.nanmean(probs_by_index, axis = 0)

    print(model_name + " average predicted final theta = " + str(pred_theta))

def figure3_iii(model, model_name, seed = 0, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1):

    np.random.seed(seed)
    
    # set up support set
    n_partitions = 20
    n_marbles_per_bag = 20
    n_marbles = n_partitions * n_marbles_per_bag
    n_sets = 2
    color_final_bag = 0

    # -------------- train for n_partition number of bags ----------------------- #
    
    train_inputs = [[j] for i in range(n_partitions) for j in [i]*n_marbles_per_bag]
    train_inputs.append([n_partitions])

    labels, proportions = sample_marbles_2(seed, n_sets = 1, n_marbles = n_marbles, n_theta = n_partitions)
    labels = labels[0]
    proportions = proportions[0]
    labels = np.append(labels, color_final_bag)
    proportions = np.append(proportions, color_final_bag)

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(labels).unsqueeze(1),  
    "train_batch_size" : 1}

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)
    
    temp_model, output_arr = simple_train_model_fig3(temp_model, training_batch, lr=lr, epochs=epochs, vary_train_batch_size=vary_train_batch_size)

    # -------------- train on single BLANK marble ----------------------- #
    
    batch = {"input_ids" : torch.FloatTensor([[n_partitions]])}
    output_final = temp_model(batch)["probs"].detach().numpy()

    # -------------- write results -------------------------------------- #
    filename = "figure3_iii_" + model_name +  "_rerun.csv"

    with open(filename, 'a', newline='') as file:
        for i in range(len(train_inputs)):

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            data = [model_name, current_time, i, train_inputs[i][0], int(labels[i]), output_arr[i][0][0], output_arr[i][0][1], proportions[int(i/20)]] 
            writer = csv.writer(file) 
            writer.writerow(data)

        data = [model_name, current_time, n_marbles+1, n_partitions, color_final_bag, output_final[0][0], output_final[0][1], color_final_bag] 
        writer = csv.writer(file)
        writer.writerow(data)

    return output_final

def figure3_iii_n_runs(model, model_name, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs = 100):
    probs_by_index = np.zeros((n_runs, 2))
    
    for i in range(n_runs):
        output = figure3_iii(model, model_name, lr=lr, seed = i, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, epochs=epochs)
        probs_by_index[i] = output[0]
        print(probs_by_index[i])

    pred_theta = np.nanmean(probs_by_index, axis = 0)

    print(model_name + " average predicted final theta = " + str(pred_theta))