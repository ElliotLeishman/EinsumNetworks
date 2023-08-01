# Test the generative ability of a PC prior, using FID to compare with test set.
import my_utils, utils, metrics
import torch
from EinsumNetwork import Graph, EinsumNetwork
import metrics
import datasets
import time
import numpy as np

# Start off by comparing each distribution type
# Train the models
#my_utils.train_model(f'../generative/Binomial/', None, EinsumNetwork.BinomialArray, K=7, pd_pieces = [4], fashion_mnist = False, num_epochs = 10, batch_size = 100, online_em_frequency = 1, online_em_stepsize = 0.05)
#my_utils.train_model(f'../generative/Normal/', None, EinsumNetwork.NormalArray, K=7, pd_pieces = [4], fashion_mnist = False, num_epochs = 10, batch_size = 100, online_em_frequency = 1, online_em_stepsize = 0.05)
#my_utils.train_model(f'../generative/Categorical/', None, EinsumNetwork.CategoricalArray, K=7, pd_pieces = [4], fashion_mnist = False, num_epochs = 10, batch_size = 100, online_em_frequency = 1, online_em_stepsize = 0.05)

# Get 2048 samples from each model
#my_utils.sampling(2048, '../generative/Binomial/', sample_dir = '../generative/samples/Binomial/1', save = True)
#my_utils.sampling(2048, '../generative/Normal/', sample_dir = '../generative/samples/Normal/1', save = True)
#my_utils.sampling(2048, '../generative/Categorical/', sample_dir = '../generative/samples/Categorical/1', save = True)

# Calculate the mean and variance of the datasets

# Load in test data
train_x, train_labels, test_x, test_labels = datasets.load_mnist()

## FID comparison
# Initialise
FID_binomial = torch.zeros(16)
FID_normal = torch.zeros(16)
FID_categorical = torch.zeros(16)
FID_binomial_1 = torch.zeros(16)
FID_normal_1 = torch.zeros(16)
FID_categorical_1 = torch.zeros(16)

# Load images
images_bin = my_utils.load_images('../generative/samples/Binomial/', grey_scale = False).type(torch.uint8)
images_nor = my_utils.load_images('../generative/samples/Normal/', grey_scale = False).type(torch.uint8)
images_cat = my_utils.load_images('../generative/samples/Categorical/', grey_scale = False).type(torch.uint8)
train_x = torch.reshape(torch.from_numpy(train_x).type(torch.uint8),(-1,1,28,28)).repeat(1, 3, 1, 1)
test_x = torch.reshape(torch.from_numpy(test_x).type(torch.uint8),(-1,1,28,28)).repeat(1, 3, 1, 1)

print(train_x.shape)
print(images_bin[1*128:(1+1)*128,:].shape)




for i in range(16):
    start = time.time()

    FID_binomial[i] = metrics.FID(test_x[i*128:(1+i)*128,:],images_bin[i*128:(1+i)*128,:])
    FID_binomial_1[i] = metrics.FID(test_x[i*128:(1+i)*128],train_x[i*128:(1+i)*128])
    print('bin')
    FID_normal[i] = metrics.FID(test_x[i*128:(1+i)*128],images_nor[i*128:(1+i)*128])
    FID_normal_1[i] = metrics.FID(test_x[i*128:(1+i)*128],train_x[i*128:(1+i)*128])
    print('nor')
    FID_categorical[i] = metrics.FID(test_x[i*128:(1+i)*128],images_cat[i*128:(1+i)*128])
    FID_categorical_1[i] = metrics.FID(test_x[i*128:(1+i)*128],train_x[i*128:(1+i)*128])
    print('cat')

    print(f'{i} ended in {np.round(time.time()-start, 2)} seconds.')

print(f'FID values for Binomial: {FID_binomial} and {FID_binomial_1}')
print(f'FID values for Normal: {FID_normal} and {FID_normal_1}')
print(f'FID values for Categorical: {FID_categorical} and {FID_categorical_1}')



    
