import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Test to see if I can get a PC working with MNIST


#Setting the parameters
############################################################################
fashion_mnist = False

# Choose exponential family 
exponential_family = EinsumNetwork.BinomialArray

# Chooses which class we want test on
classes = [7]

# Number of sums/input distributions
K = 10

# Choose structure
#structure = 'poon-domingos'
structure = 'binary-trees'

# 'poon-domingos

# pd_num_peices has some effect on the number of layers in the PC
pd_num_pieces = [5,7]
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]

# Presume these since MNIST is 28x28
width = 28
height = 28

# 'binary-trees'
depth = 3
num_repetitions = 20

# Training Parameters
num_epochs = 5
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05

############################################################################

# What is this doing
exponential_family_args = None
if exponential_family == EinsumNetwork.BinomialArray:
    exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
    exponential_family_args = {'K': 256}
if exponential_family == EinsumNetwork.NormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

# Load in the data - just using MNIST
train_x, train_labels, test_x, test_labels = datasets.load_mnist()

# validation split
valid_x = train_x[-10000:, :]
train_x = train_x[:-10000, :]
valid_labels = train_labels[-10000:]
train_labels = train_labels[:-10000]


# Test, train, validation split
# pick the selected classes
if classes is not None:
    train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
    valid_x = valid_x[np.any(np.stack([valid_labels == c for c in classes], 1), 1), :]
    test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

train_x = torch.from_numpy(train_x).to(torch.device(device))
valid_x = torch.from_numpy(valid_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))

# Make EinsumNetwork
######################################

# Sets the structure of PC
if structure == 'poon-domingos':
    pd_delta = [[height / d, width / d] for d in pd_num_pieces]
    graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
elif structure == 'binary-trees':
    graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)
else:
    raise AssertionError("Unknown Structure")

args = EinsumNetwork.Args(

        num_var=train_x.shape[1],
        num_dims=1,
        num_classes=1,
        num_sums=K,
        num_input_distributions=K,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        online_em_frequency=online_em_frequency,
        online_em_stepsize=online_em_stepsize)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)

# Train
######################################

train_N = train_x.shape[0]
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]

for epoch_count in range(num_epochs):

    ##### evaluate
    einet.eval()
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
    valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
    test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
    print("[{}]   train LL {}   valid LL {}   test LL {}".format(
        epoch_count,
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))
    einet.train()
    #####

    idx_batches = torch.randperm(train_N, device=device).split(batch_size)

    total_ll = 0.0
    for idx in idx_batches:
        batch_x = train_x[idx, :]
        outputs = einet.forward(batch_x)
        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()
        log_likelihood.backward()

        einet.em_process_batch()
        total_ll += log_likelihood.detach().item()

    einet.em_update()

if fashion_mnist:
    model_dir = '../models/einet/demo_fashion_mnist/'
    samples_dir = '../samples/demo_fashion_mnist/'
else:
    model_dir = '../models/einet/demo_mnist/'
    samples_dir = '../samples/demo_mnist/'
utils.mkdir_p(model_dir)
utils.mkdir_p(samples_dir)