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

def FID(imgs_true, imgs_false, features = 2048):

    fid = FrechetInceptionDistance(feature=features)

    # Load images as tensors
    #imgs_true = load_images(dir_real).type(torch.uint8)[:127]
    #imgs_false = load_images(dir_gen).type(torch.uint8)[:127]

    #print('Images have been loaded in.')

    # Calculate and print the FID distance
    fid.update(imgs_false, real=False)
    print('FID object updated with generated images.')
    fid.update(imgs_true, real=True)
    print('FID object updated with true images.')


    return fid.compute()

def MSE(clean, noisy):
    return torch.sum(torch.square(clean - noisy), dim = (2,3))/784

def PSNR(clean,noisy):
    return -10 * torch.log10(MSE(clean, noisy))