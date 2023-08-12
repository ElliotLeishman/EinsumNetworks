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



# Expectation of the whole mnist prior
utils.mkdir_p('../expectation/demo_mnist/2/')

#einet = torch.load('../models/einet/demo_mnist/einet.mdl')
#my_utils.image_expectation(einet, 784, 28, K = 10, save = True, save_dir = '../expectation/demo_mnist/2/image_0.png')

# Set the noise
sigma = 0.01

# Make 100 noisy 5's
my_utils.gaussian_noise('../blur/',save = True, save_dir = '../blur2/2')


for i in range(100):
    noisy_image = my_utils.load_images('../blur2/', True)[i,:,:,:]
    my_utils.denoising_expectation('../models/einet/demo_mnist_5/einet.mdl',noisy_image, 0.01, 784, 28, K = 15, save = True, save_dir = f'../expectation/demo_mnist_5/1/denoised_{i}.png')