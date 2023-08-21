# Contains the functions for Inverse problems
import numpy as np
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import utils
import datasets
from EinsumNetwork import Graph, EinsumNetwork
import time
import my_utils


def image_expectation(model, num_pixels, batch_size, K = 15, gaussian = True, means = None, save = False, save_dir = None):
    '''
    This function works for non-Gaussian distributions, but not sure that the lower-level
    functions are compatible with them yet.

    Best if batch_size divides num_pixels - though shouldn't be hard to fix.
    '''

    dist_layer = model.einet_layers[0]
    
    # Get distribution means
    if means is not None:
        means = means
    elif gaussian:
        means = dist_layer.ef_array.params.squeeze()[:,:,0].unsqueeze(2)
    else:
        means = dist_layer.ef_array.params.squeeze().unsqueeze(2)

    # Initialise expectation tensor
    expectations = torch.zeros(784)

    for batch_no in range(num_pixels // batch_size):
        #print(f'Batch {batch_no}')
        x = torch.zeros((batch_size, num_pixels, K, 1))

        for idx in range(batch_size):
            pixel = batch_size * batch_no + idx
            x[idx,pixel,:] = means[pixel,:]
        
        expectations[batch_no*batch_size:(batch_no + 1)*batch_size] = model.expectation(x).detach().squeeze()

    if save and save_dir is not None:
        #utils.mkdir_p(save_dir)
        utils.save_image_stack(torch.reshape(expectations,(1,28,28)),\
         1, 1, filename = save_dir, margin_gray_val=0.)
        #print(f'Images saved to {save_dir}')
    elif save:
        print('Need name of directory to save images to...')

    return expectations

#print(image_expectation('../models/einet/demo_mnist_5/einet.mdl', 784, 28, 15, True))
#print(image_expectation('../models/einet/demo_mnist_3/einet.mdl', 784, 28, 15, False))


def get_variance(phi, dist_layer):
    '''Only works for NormalArray'''
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

# gaussian_product(phi, torch.ones(784,15,1), 2)

def denoising_expectation(model_file, noisy_image, epsilon, num_pixels, batch_size, K = 15, gaussian = True, save = False, save_dir = None):
    ''' Makes use of newly improved expectation function'''

    # Get model
    einet = torch.load(model_file)
    dist_layer = einet.einet_layers[0]

    # Format noisy image
    y = torch.reshape(noisy_image,(1,784)).squeeze()
    y = torch.transpose(y.repeat(K,1),0,1).unsqueeze(2)
    #print(y.shape)

    # Distibution parameters
    phi = dist_layer.ef_array.params
    mean, var = gaussian_product(phi, y, epsilon, dist_layer)

    return image_expectation(einet, num_pixels, batch_size, K = K, gaussian = gaussian, means = mean, save = save, save_dir = save_dir)
    

def depainting(model, A, noisy_image, K=15, sigma = 0.01, img_name = 0, save_dir = None):
    # sigma is actually variance so should be sigma^2
    noisy_image = torch.reshape(noisy_image, (1, 784))

    # Format noisy image
    y = torch.reshape(noisy_image,(1,784)).squeeze()
    y = torch.transpose(y.repeat(K,1),0,1).unsqueeze(2)

    # Prior means
    einet = model
    dist_layer = einet.einet_layers[0]
    phi = dist_layer.ef_array.params
    prior_means = phi[:,:,:,0]
    comb_means, var = gaussian_product(phi, y, sigma, dist_layer)
    #print(post_means.shape)

    # Initialise the mean vector and the multiplier
    means = torch.zeros(784,K,1)
    multiplier = torch.ones(784)

    # Reshape A
    A = torch.reshape(A,(1,784))

    for i in range(A.shape[1]):
        if A[:,i] == 1:
            #x = torch.zeros(1,784, 15, 1)
            means[i, :] = comb_means[i]

        elif A[:,i] == 0:
            #x = torch.zeros(1,784, 15, 1)
            means[i, :] = prior_means[i]

            # Deterministic part - multiplier
            #multiplier[i]= torch.exp(-0.5 * (noisy_image[:,i]+0.5)**2 / sigma)

        else:
            raise

    result = image_expectation(einet, 784, 28, K=K, means = means)

    return multiplier * result
