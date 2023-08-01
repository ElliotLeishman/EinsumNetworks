# This script contains functions to apply forward models
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

def gaussian_noise(img_dir, gs = True, sigma = 0.05, save = False, save_dir = None):
    
    # Load the images and store them as a tensor
    x = load_images(img_dir, grey_scale = gs)

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