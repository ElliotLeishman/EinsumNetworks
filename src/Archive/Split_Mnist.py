import os
import numpy as np
import torch
import datasets
import utils
import torchvision.transforms as T
from PIL import Image

'''
This code will take the (Fashion) MNIST dataset and split it into two parts
containing the classes the that are asked for. 
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fashion_mnist = False

# Choose the classes we want
classes = [7]

# Get data
if fashion_mnist:
    train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
else:
    train_x, train_labels, test_x, test_labels = datasets.load_mnist()

# Pick the selected classes
if classes is not None:
    train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
    test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

# Convert to pytorch tensors
train_x = torch.from_numpy(train_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))

# Split the training set into 2
split = len(train_x) // 2
train_x1 = train_x[:split]
train_x2 = train_x[split:]

# Set/ make directory
if fashion_mnist:
    part_1 = f'../split_datasets/fashion3/part_1/1'
    part_2 = f'../split_datasets/fashion3/part_2/1'
else:
    part_1 = f'../split_datasets/mnist7/part_1/1'
    part_2 = f'../split_datasets/mnist7/part_2/1'

utils.mkdir_p(part_1)
utils.mkdir_p(part_2)

print(train_x1.size())

# unnecessary? - define a transform to convert a tensor to PIL image
transform = T.ToPILImage()

# Save train_x1 in part_1 directory
for i in range(len(train_x1)):
    utils.save_image_stack(torch.reshape(train_x1[i],(1,28,28)), 1, 1, os.path.join(part_1, f'image_{i}.png'), margin_gray_val=0.)


# Save train_x2 in part_2 directory
for i in range(len(train_x2)):
    utils.save_image_stack(torch.reshape(train_x2[i],(1,28,28)), 1, 1, os.path.join(part_2, f'image_{i}.png'), margin_gray_val=0.)
