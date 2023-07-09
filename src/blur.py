'''
This script will first take an image and apply Guassian noise to it
'''

import numpy as np
import torch
import utils
from my_utils import*
import os



x = load_images("../blur/", grey_scale = True).squeeze()
shape = x.size()
x = torch.flatten(x, start_dim = 1)
d = x.size()[1]

# Make noise
sigma = 0.05
means = np.zeros(d)
cov = sigma * np.eye(d)

w = np.random.multivariate_normal(means, cov, shape[0])

y = x + w
print(y)

dir = f'../blur2/2'
utils.mkdir_p(dir)


for i in range(len(y)):
    utils.save_image_stack(torch.reshape(y[i],(1,28,28)), 1, 1, os.path.join(dir, f'noise_image_{i}.png'), margin_gray_val=0.)

print('Images Saved')