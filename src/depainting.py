# Should get rid of these eventually
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
import expectation

images = my_utils.load_images('../inpainting/', grey_scale = True)
A = images[0,:]
A = torch.reshape(A, (1, 784))#[:,:600]
noisy_image = images[1,:]
print(A.shape)

def depainting(A, noisy_image, sigma = 0.01, img_no = 0):
    # sigma is actually variance so should be sigma^2
    noisy_image = torch.reshape(noisy_image, (1, 784)) 

    # Format noisy image
    y = torch.reshape(noisy_image,(1,784)).squeeze()
    y = torch.transpose(y.repeat(15,1),0,1).unsqueeze(2)

    # Prior means
    einet = torch.load('../models/einet/demo_mnist_5/einet.mdl')
    dist_layer = einet.einet_layers[0]
    phi = dist_layer.ef_array.params
    means = phi[:,:,:,0]
    post_means, var = expectation.gaussian_product(phi, y, sigma, dist_layer)
    print(post_means.shape)

    # Initialise the expectation vector
    expectations = torch.zeros(784)



    for i in range(A.shape[1]):
        if A[:,i] == 1:

            x = torch.zeros(1,784, 15, 1)
            x[0, i, :] = post_means[i]

            expectations[i] = einet.expectation(x)
            

        elif A[:,i] == 0:

            x = torch.zeros(1,784, 15, 1)
            x[0, i, :] = means[i]

            # Deterministic part
            exp = torch.exp(-0.5 * noisy_image[:,i]**2 / sigma)

            expectations[i] = exp * einet.expectation(x)
            print(i)

        else:
            raise
    expectations = expectations.detach()
    print(expectations)
    save_dir = '../depainted/1/'
    utils.mkdir_p(save_dir)
    utils.save_image_stack(torch.reshape(expectations,(1,28,28)),\
             1, 1, os.path.join(save_dir, f'image_{img_no}.png'),\
             margin_gray_val=0.)

    

# for i in range(10):
#     images = my_utils.load_images('../inpainting/', grey_scale = True)
#     noisy_image = images[i+1,:]
#     depainting(A, noisy_image, img_no = i)


def depainting_2(A, noisy_image, sigma = 0.01, img_no = 0):
    # sigma is actually variance so should be sigma^2
    noisy_image = torch.reshape(noisy_image, (1, 784))

    # Format noisy image
    y = torch.reshape(noisy_image,(1,784)).squeeze()
    y = torch.transpose(y.repeat(15,1),0,1).unsqueeze(2)

    # Prior means
    einet = torch.load('../models/einet/demo_mnist_5/einet.mdl')
    dist_layer = einet.einet_layers[0]
    phi = dist_layer.ef_array.params
    prior_means = phi[:,:,:,0]
    comb_means, var = expectation.gaussian_product(phi, y, sigma, dist_layer)
    #print(post_means.shape)

    # Initialise the mean vector and the multiplier
    means = torch.zeros(784,15,1)
    multiplier = torch.ones(784)

    for i in range(A.shape[1]):
        if A[:,i] == 1:
            #x = torch.zeros(1,784, 15, 1)
            means[i, :] = comb_means[i]

        elif A[:,i] == 0:
            #x = torch.zeros(1,784, 15, 1)
            means[i, :] = prior_means[i]

            # Deterministic part - multiplier
            multiplier[i]= torch.exp(-0.5 * noisy_image[:,i]**2 / sigma)

        else:
            raise

    return multiplier * expectation.image_expectation(einet, 784, 28, means = means, save = True, save_dir = f'../depainted/1/image_{img_no}.png')


for i in range(10):
    images = my_utils.load_images('../inpainting/', grey_scale = True)
    noisy_image = images[i+1,:]
    depainting_2(A, noisy_image, img_no = i)


