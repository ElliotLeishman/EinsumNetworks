import numpy as np
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance


def load_images(dir, grey_scale = False):
    '''
    Takes a directory containing images and returns a tensor of those images
    '''

    # Define the transformations
    if grey_scale:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    # Define the image folder and data loader
    img_folder= ImageFolder(dir, transform = transform)
    img_loader = DataLoader(img_folder, batch_size=None)

    images = torch.zeros(len(img_loader), *img_folder[0][0].shape)

    for idx, samples in enumerate(img_folder):
        images[idx] = samples[0]

    return images#.squeeze()


fid = FrechetInceptionDistance(feature=64)

# Load images as tensors
imgs_true = load_images('../split_datasets/mnist3/').type(torch.uint8)[:100]
imgs_false = load_images('../samples/demo_mnist_3/').type(torch.uint8)[:100]


print(type(imgs_true), imgs_true.size(), imgs_true.dtype)
#breakpoint()
# Calculate and print the FID distance
fid.update(imgs_true, real=True)
print('test')
fid.update(imgs_false, real=False)
fid.compute()
