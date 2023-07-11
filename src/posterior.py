import numpy as np
import torch
import os
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils
import my_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load in the model I have already trained
#model_file = os.path.join('../models/einet/demo_mnist_5/', "einet.mdl")
#einet = torch.load(model_file)

#print(einet)
#dist_layer = einet.einet_layers[0]
#phi = dist_layer.ef_array.params
#print(phi[:,:,:,1])
#print(phi.shape, phi)

def get_variance(phi, dist_layer):

    theta = dist_layer.ef_array.expectation_to_natural(phi)
    variance = theta[:,:,:,1] * -2
    variance = torch.reciprocal(variance)

    return variance


#print(get_variance(phi))


def gaussian_product(phi, y, epsilon, dist_layer):
    '''
    Returns the mean and variance of the product of two Gaussians
    means and variance 1's refer to statistics of distribution
    means and variance 2's refer to statistics of denoising Gaussian
    '''

    # Get statistics of the distribtuion units
    variance1 = get_variance(phi, dist_layer)
    mean1 = phi[:,:,:,0]
    
    # Calculate combined mean
    mean_num = epsilon * mean1 + torch.mul(y, variance1)
    denom = torch.add(variance1, epsilon)
    mean = mean_num / denom

    # Calculate combined variances
    var_num = variance1 * epsilon
    var = var_num / denom
    
    return mean, var

#gaussian_product(phi, torch.ones(784,15,1), 2)

def denoising_expectation(num_pixels, y, epsilon):

    # Load model and retrieve parameters
    model_file = os.path.join('../models/einet/demo_mnist_5/', "einet.mdl")
    einet = torch.load(model_file)
    dist_layer = einet.einet_layers[0]
    phi = dist_layer.ef_array.params

    # Initialise the expectations
    expectations = torch.zeros(784)

    mean, var = gaussian_product(phi, y, epsilon, dist_layer)
    print(mean.shape)

    for pix in range(num_pixels):
        mean2 = torch.zeros((1,784,15,1))
        mean2[:,pix,:,:] = mean[pix,:,:]
        expectations[pix] = einet.expectation(mean2)
        print(pix)

    return expectations.detach()

#print(denoising_expectation(100, torch.ones(784,15,1), 2))

#image = denoising_expectation(500, torch.ones(784,15,1), 2)

image = my_utils.image_expectation(784)
exp_dir = '../expectation/demo_mnist_5/1/'
utils.mkdir_p(exp_dir)
utils.save_image_stack(torch.reshape(image,(1,28,28)), 1, 1, os.path.join(exp_dir, f'exp_image_5.png'), margin_gray_val=0.)
