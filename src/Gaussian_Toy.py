import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
###############################################################
'''
Want to get an EiNet working on randomly generated multivariate Gaussian data
'''

# Generate data
N = 10000 # number of samples
D = 5 # dimension

means = np.random.standard_normal(D)
covs = np.ones(D)

data = np.transpose(np.random.multivariate_normal(means, np.diag(covs), size = N))

train_x = torch.from_numpy(data).to(torch.device(device))

print(train_x, train_x.shape)

# Parameters
############################################################################
# Want by leaves to be univariate Gaussians 
exponential_family = EinsumNetwork.NormalArray
exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}


# Number of Gaussians in each leaf
K = 10

structure = 'poon-domingos'


# 'poon-domingos'
pd_num_pieces = [4]
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]
width = 28
height = 28

# 'binary-trees'
depth = 3
num_repetitions = 20


num_epochs = 5
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05
############################################################################


graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)

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

##################################################################################
train_N = train_x.shape[0]

for epoch_count in range(num_epochs):

    ##### evaluate
    einet.eval()
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
    print("[{}]   train LL {}".format(
        epoch_count,
        train_ll / train_N))
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