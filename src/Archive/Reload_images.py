import numpy as np
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set number of datasets to compare
n = 2

data_dir1 = f'../split_datasets/mnist3/part_1/'
data_dir2 = f'../samples/demo_mnist_3/'


transformations = transforms.Compose([
    transforms.ToTensor()
])

set_1 = datasets.ImageFolder(data_dir1, transform = transformations)
set1_loader = torch.utils.data.DataLoader(set_1, batch_size=None)

set_2 = datasets.ImageFolder(data_dir2, transform = transformations)
set2_loader = torch.utils.data.DataLoader(set_2, batch_size=None)

sums_1 = np.zeros(len(set1_loader))
sums_2 = np.zeros(len(set2_loader))



for batch_idx, samples in enumerate(set1_loader):
    sums_1[batch_idx] = torch.sum(samples[0])

for batch_idx, samples in enumerate(set2_loader):
    sums_2[batch_idx] = torch.sum(samples[0])

print(sums_1)


print(f'Mean of dataset 1 is: {np.mean(sums_1)}, variance is: {np.var(sums_1)}')
print(f'Mean of dataset 2 is: {np.mean(sums_2)}, variance is: {np.var(sums_2)}')