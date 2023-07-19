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
    print(f'mean size is {mean_num.shape}')
    denom = torch.add(variance1, epsilon)
    mean = mean_num / denom
    print(f'mean size is {mean.shape}')


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

    y = torch.transpose(y.repeat(15,1),0,1).unsqueeze(2)


    mean, var = gaussian_product(phi, y, epsilon, dist_layer)
    print(mean.shape)

    for pix in range(num_pixels):
        mean2 = torch.zeros((1,784,15,1))
        mean2[:,pix,:,:] = mean[pix,:,:]
        expectations[pix] = einet.expectation(mean2)
        print(pix)

    return expectations.detach()

# #print(denoising_expectation(100, torch.ones(784,15,1), 2))

# #image = denoising_expectation(500, torch.ones(784,15,1), 2)

# #image = my_utils.image_expectation(784)


# #noisy_images = my_utils.load_images('../blur2/', True)[0,:,:,:]
# #noisy_images = torch.reshape(noisy_images,(1,784)).squeeze()
# #print(noisy_images)


# #image = denoising_expectation(620, noisy_images, 0.05)
# #print(image)

# #exp_dir = '../expectation/demo_mnist_5/1/'
# #utils.mkdir_p(exp_dir)
# #utils.save_image_stack(torch.reshape(image,(1,28,28)), 1, 1, os.path.join(exp_dir, f'exp_image_6.png'), margin_gray_val=0.)

# # x = torch.ones(784)
# # print(f'x is size {x.shape}')
# # x = torch.transpose(x.repeat(15,1),0,1).unsqueeze(2)
# # print(f'x is size {x.shape}')



# ###################################################################################################
# # Try and calculate expectation using batches

# batch_size = 28
# num_pixels = 784
# K = 15


# model_file = os.path.join('../models/einet/demo_mnist_5/', "einet.mdl")
# einet = torch.load(model_file)
# dist_layer = einet.einet_layers[0]
# phi = dist_layer.ef_array.params.squeeze()[:,:,0].unsqueeze(2) # hacky
# print(phi.shape)

# expectations = torch.zeros(784)

# #batch_no = 0
# print(num_pixels // batch_size)
# for batch_no in range(num_pixels // batch_size):
#     print(batch_no)
#     x = torch.zeros((batch_size, num_pixels, K, 1))
#     for idx in range(batch_size):

#         pixel = batch_size * batch_no + idx
#         x[idx,pixel,:] = phi[pixel,:]
#     expectations[batch_no*batch_size:(batch_no + 1)*batch_size] = einet.expectation(x).detach().squeeze()
# print(expectations)


def image_expectation(model_file, num_pixels, batch_size,K = 15, gaussian = True):
    '''
    This function works for non-Gaussian distributions, but not sure that the lower-level
    functions are compatible with them yet.

    Best if batch_size divides num_pixels - though shouldn't be hard to fix.
    '''

    einet = torch.load(model_file)
    dist_layer = einet.einet_layers[0]
    
    # Get distribution means
    if gaussian:
        means = dist_layer.ef_array.params.squeeze()[:,:,0].unsqueeze(2)
    else:
        means = dist_layer.ef_array.params.squeeze().unsqueeze(2)
    print(means.shape)

    # Initialise expectation tensor
    expectations = torch.zeros(784)

    for batch_no in range(num_pixels // batch_size):
        print(f'Batch {batch_no}')
        x = torch.zeros((batch_size, num_pixels, K, 1))

        for idx in range(batch_size):
            pixel = batch_size * batch_no + idx
            x[idx,pixel,:] = means[pixel,:]
        
        expectations[batch_no*batch_size:(batch_no + 1)*batch_size] = einet.expectation(x).detach().squeeze()

    return expectations


print(image_expectation('../models/einet/demo_mnist_5/einet.mdl', 784, 28, 15, True))
#print(image_expectation('../models/einet/demo_mnist_3/einet.mdl', 784, 28, 15, False))


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


