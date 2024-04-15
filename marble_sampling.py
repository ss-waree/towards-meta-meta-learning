import numpy as np
import torch


def sample_marbles(seed, n_sets, n_marbles, alpha, beta_0, beta_1):

    torch.manual_seed(seed)
    lambda_param = 1
    n_colors = 2
    beta = torch.tensor([beta_0, beta_1])
    marbles = np.zeros((n_sets,n_marbles))
    theta = torch.distributions.Dirichlet(alpha * beta).sample().numpy()
    for i in range(n_sets):
        num0 = round(n_marbles * theta[0])
        marbles[i,:num0] = 0
        marbles[i,num0:] = 1
        # Generate a random permutation of indices
        shuffled_indices = torch.randperm(n_marbles).numpy()
        # Shuffle the marbles using these indices
        marbles[i] = marbles[i][shuffled_indices]

    return marbles

''' 
this creates a collection of marbles from multiple theta but the
same alpha and beta 
'''
def sample_marbles_2(seed, n_sets, n_marbles, n_theta):

    torch.manual_seed(seed)
    lambda_param = 1

    # we need alpha to be non-zero positive so we can sample from theta
    n_colors = 2
    alpha = 0.0
    while alpha == 0: 
        alpha = torch.distributions.Exponential(lambda_param).sample()

    beta = torch.distributions.Dirichlet(torch.ones(n_colors)).sample()

    # training and testing need to be generated from the same theta
    marbles = np.zeros((n_sets,n_marbles))

    n_marbles_per_bag = int(n_marbles/n_theta)
    marbles_one_partition = np.zeros((n_sets, n_marbles_per_bag))

    proportions = np.zeros((n_sets,n_theta))

    for t in range(n_theta):
        theta = torch.distributions.Dirichlet(abs(alpha) * beta).sample().numpy()
        for i in range(n_sets):
            num0 = round(n_marbles_per_bag * theta[0])
            marbles_one_partition[i,:num0] = 0
            marbles_one_partition[i,num0:] = 1
            # Generate a random permutation of indices
            shuffled_indices = torch.randperm(n_marbles_per_bag).numpy()
            # Shuffle the marbles using these indices
            marbles[i,n_marbles_per_bag*t:n_marbles_per_bag*(t+1)] = marbles_one_partition[i][shuffled_indices]
            proportions[i,t] = 1 - num0/n_marbles_per_bag

    return marbles, proportions

''' 
this creates a collection of marbles from multiple theta but the
same alpha and beta 
'''
def sample_marbles_3(seed, partition_sizes_arr):

    torch.manual_seed(seed)
    lambda_param = 1

    # we need alpha to be non-zero positive so we can sample from theta
    n_colors = 2
    alpha = 0.0
    while alpha == 0: 
        alpha = torch.distributions.Exponential(lambda_param).sample()

    beta = torch.distributions.Dirichlet(torch.ones(n_colors)).sample()

    # print("alpha", alpha)
    # print("beta", beta)

    # training and testing need to be generated from the same theta
    marbles = []

    for size in partition_sizes_arr:
        theta = torch.distributions.Dirichlet(torch.abs(alpha) * beta).sample()
        # print("theta", theta)
        num0 = int(round(size * theta[0].item()))
        marbles_one_partition = torch.zeros(size, dtype=torch.int64)
        marbles_one_partition[num0:] = 1
        # Generate a random permutation of indices
        shuffled_indices = torch.randperm(size)
        # Shuffle the marbles using these indices
        marbles.append(marbles_one_partition[shuffled_indices])

    return marbles

''' 
this creates a collection of marbles from multiple theta but the
same alpha and beta 
'''
def sample_marbles_transformers(seed, alpha, beta, n_bags, n_marbles_per_bag):

    torch.manual_seed(seed)

    # print("alpha", alpha)
    # print("beta", beta

    # for now we'll do evenly sized bags so we don't have to deal with padding
    marbles = torch.zeros((n_bags,n_marbles_per_bag))

    for bag in range(n_bags):
        theta = torch.distributions.Dirichlet(torch.abs(alpha) * beta).sample()
        # print("theta", theta)
        num0 = int(round(n_marbles_per_bag * theta[0].item()))
        marbles_one_bag = torch.zeros(n_marbles_per_bag, dtype=torch.int64)
        marbles_one_bag[num0:] = 1
        # Generate a random permutation of indices
        shuffled_indices = torch.randperm(n_marbles_per_bag)
        # Shuffle the marbles using these indices
        marbles[bag] = marbles_one_bag[shuffled_indices]

    return marbles

''' 
this creates a collection of marbles from multiple theta but the
same alpha and beta, VARIABLE LENGTH BAGS
'''
def sample_marbles_transformers_2(seed, alpha, beta, n_bags, n_marbles_each_bag, max_marbles):

    torch.manual_seed(seed)

    marbles = torch.zeros((n_bags,max_marbles))

    for bag in range(n_bags):
        theta = torch.distributions.Dirichlet(torch.abs(alpha) * beta).sample()
        # print("theta", theta)
        num0 = int(round(n_marbles_each_bag[bag] * theta[0].item()))
        marbles_one_bag = torch.zeros(n_marbles_each_bag[bag], dtype=torch.int64)
        marbles_one_bag[num0:] = 1
        # Generate a random permutation of indices
        shuffled_indices = torch.randperm(n_marbles_each_bag[bag])
        # Shuffle the marbles using these indices
        marbles_without_padding = marbles_one_bag[shuffled_indices]

        # Pad the marbles_without_padding tensor with 2s
        padding_size = max_marbles - n_marbles_each_bag[bag]
        padding_tensor = torch.full((padding_size,), 2, dtype=torch.int64)
        marbles_with_padding = torch.cat((marbles_without_padding, padding_tensor))
                
        marbles[bag] = marbles_with_padding

    return marbles

# def sample_marbles_3(seed, n_marbles):

#     torch.manual_seed(seed)
#     lambda_param = 1
#     n_colors = 2

#     alpha = 0.0
#     while alpha == 0: 
#         alpha = torch.distributions.Exponential(lambda_param).sample()

#     beta = torch.distributions.Dirichlet(torch.ones(n_colors)).sample()
#     marbles = np.zeros(n_marbles)

#     theta = torch.distributions.Dirichlet(alpha * beta).sample().numpy()
#     num0 = round(n_marbles * theta[0])
#     marbles[:num0] = 0
#     marbles[num0:] = 1
#     # Generate a random permutation of indices
#     shuffled_indices = torch.randperm(n_marbles).numpy()
#     # Shuffle the marbles using these indices
#     return marbles[shuffled_indices]

if __name__ == "__main__":
    print(sample_marbles_2(seed = 29, n_sets=2, n_marbles=27, n_theta=3))
    # print("DNF DATASET")
    # create_dnf_dataset = dnf_dataset(4)
    # for i in range(1):
    #     print(create_dnf_dataset(i))
    #     print("")
    
    # all black or white marbles in a train-test set
    # rand_choice = np.random.randint(2)
    # if rand_choice == 0:
    #     return np.zeros((n_sets,n_marbles))
    # else:
    #     return np.ones((n_sets,n_marbles))

# Display the sampled values with 2 decimal points
# print(f"Sampled alpha: \n{alpha.item():.2f}")
# print(f"Sampled beta: \n{np.array2string(beta.numpy(), formatter={'float_kind':lambda x: f'{x:.2f}'})}")
# print(f"Sampled theta: \n{np.array2string(theta_arr, formatter={'float_kind':lambda x: f'{x:.2f}'})}")
# print(f"Generated marbles: \n{marbles}")

