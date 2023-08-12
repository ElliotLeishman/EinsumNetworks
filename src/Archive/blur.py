import numpy as np
import my_utils, utils, metrics, ULA
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os
from torchmetrics.image import PeakSignalNoiseRatio
import time as time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Trying to blur some images
###################################
# Load some images
x = my_utils.load_images('../experiments/prior/mnist_[7]/test', grey_scale = True)[:100,:].squeeze()
# print(x.shape)

# # Blur kernel
# kernel_len = [5,5] # This value controls 'blurriness'
# size = [x.shape[0],x.shape[1]]
# type_blur = "uniform"
# A, AT, AAT_norm = ULA.blur_operators(kernel_len, size, type_blur, device)

# y0 = A(x)

# utils.mkdir_p('../blur/1/')
# utils.save_image_stack(torch.reshape(y0,(1,28,28)),1,1,'../blur/1/test_1.png')


# Create a blurring function
##########################################

def blurring(images, save_dir, kernel_len = [5,5], type_blur = 'uniform', var = None, device = 'cpu'):

    # images should be of the form (n,h,w)
    shape = images.shape

    # Define the blur operator
    A, AT, AAT_norm = ULA.blur_operators(kernel_len, [shape[1],shape[2]], type_blur, device, var = var)

    # Make directory
    utils.mkdir_p(save_dir)

    # Loop over images and apply the blur operator
    for i in range(shape[0]):
        utils.save_image_stack(torch.reshape(A(images[i,:]),(1,28,28)),1,1,os.path.join(save_dir, f'blurred_{i}.png'))
    print(f'Images saved to {save_dir}')

    return A,AT,AAT_norm

blurring(x,'../blur/1/', )


