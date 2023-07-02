import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set the correct directory
fashion_mnist = False
if fashion_mnist:
    model_dir = '../models/einet/demo_fashion_mnist/'
    samples_dir = '../samples/demo_fashion_mnist/1/'
else:
    model_dir = '../models/einet/demo_mnist_3/'
    samples_dir = '../samples/demo_mnist_3/1/'
    
utils.mkdir_p(samples_dir)

model_file = os.path.join(model_dir, "einet.mdl")

# Reload model
einet = torch.load(model_file)
print("Loaded model from {}".format(model_file))
print(einet)

# Naive sampling of the model - Think this is working
num_sam = 1000
samples = einet.sample(num_samples=num_sam).cpu()

for i in range(num_sam):
    utils.save_image_stack(torch.reshape(samples[i],(1,28,28)), 1, 1, os.path.join(samples_dir, f'image_{i}.png'), margin_gray_val=0.)

