import my_utils, utils, metrics
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os
import time

# # Load a random image
# image = my_utils.load_images('../blur/', grey_scale=True)[1]

# for i in range(28,785):
#     if 784%i==0:
#         print(i)
#         start = time.time()
#         expectation.denoising_expectation('../models/einet/demo_mnist/einet.mdl', image, 0.01, 784, i, K = 10, gaussian = True, save = False, save_dir = None)
#         end = time.time()-start
#         print(f'Time taken for batch size {i} was {end} seconds.')


einet = torch.load('../models/einet/demo_mnist/einet.mdl')
print(einet)