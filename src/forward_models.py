# This script contains functions to apply forward models
import numpy as np
import my_utils, utils, metrics, ULA
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os
from torchmetrics.image import PeakSignalNoiseRatio
import time as time

def gaussian_noise(img_dir, gs = True, sigma = 0.05, save = False, save_dir = None):
    
    # Load the images and store them as a tensor
    x = my_utils.load_images(img_dir, grey_scale = gs)

    shape = x.size() # Save the shape of image tensors for later
    x = torch.flatten(x, start_dim = 1)
    d = x.size()[1]

    # Noise
    means = np.zeros(d)
    cov = sigma * np.eye(d)

    w = np.random.multivariate_normal(means, cov, shape[0])
    y = x + w 

    if save and save_dir is not None:
        utils.mkdir_p(save_dir)
        for i in range(len(y)):
            utils.save_image_stack(torch.reshape(y[i],(1,shape[2],shape[2])), 1, 1, os.path.join(save_dir, f'noise_image_{i}.png'), margin_gray_val=0.)
        print(f'Images saved to {save_dir}')
    elif save:
        print('Need name of directory to save images to...')

    return y

def create_inpainting_matrix(num_pixels, prop):

    torch.manual_seed(42)
    vector = torch.ones(num_pixels)
    vector[torch.randperm(num_pixels)[:int(num_pixels * prop)]] = 0

    return vector

def inpainting(image, prop, noise = True, sigma = 0.05, save = False, save_dir = None):

    shape = image.shape
    d = shape[2] * shape[3]
    image = torch.reshape(image, (shape[0], shape[1], d))

    A = create_inpainting_matrix(image.shape[2], prop)
    #image =  A * image
    if noise:
        gauss_noise = np.random.multivariate_normal(np.zeros(d), sigma * np.eye(d), shape[0])
        image = A * image + torch.from_numpy(gauss_noise).unsqueeze(1)
    else:
        image = A * image

    if save and save_dir is not None:
        utils.mkdir_p(save_dir)
        utils.save_image_stack(torch.reshape(A,(1,shape[2],shape[3])), 1, 1, os.path.join(save_dir, f'a_matrix.png'), margin_gray_val=0.)
        for i in range(shape[0]):
            utils.save_image_stack(torch.reshape(image[i],(1,shape[2],shape[3])), 1, 1, os.path.join(save_dir, f'inpainted_{i}.png'), margin_gray_val=0.)
        print(f'Images saved to {save_dir}')
    elif save:
        print('Need name of directory to save images to...')

    return image, A


def blurring(images, save_dir, sig2 = 0.001, kernel_len = [5,5], type_blur = 'uniform', var = None, device = 'cpu'):

    # images should be of the form (n,h,w)
    shape = images.shape

    if len(shape) == 2:
        images = images.unsqueeze(0)
        shape = images.shape

    # Define the blur operator
    A, AT, AAT_norm = ULA.blur_operators(kernel_len, [shape[1],shape[2]], type_blur, device, var = var)

    # Make directory
    utils.mkdir_p(save_dir)

    # Loop over images and apply the blur operator
    for i in range(shape[0]):
        utils.save_image_stack(torch.reshape(A(images[i,:]) + sig2 * torch.randn_like(images[i,:]),(1,28,28)),1,1,os.path.join(save_dir, f'blurred_{i}.png'))
    print(f'Images saved to {save_dir}')

    return A, AT, AAT_norm
