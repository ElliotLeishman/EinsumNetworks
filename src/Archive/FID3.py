# import torch
# _ = torch.manual_seed(123)
# from torchmetrics.image.fid import FrechetInceptionDistance
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader

# fid = FrechetInceptionDistance(feature=64)

# # Define transformation
# transform = transforms.Compose([
#     transforms.ToTensor()])

# imgs_true = ImageFolder(root='../split_datasets/mnist3/part_1/', transform=transform)
# imgs_false = ImageFolder(root='../samples/demo_mnist_3/', transform=transform)

# # Calculate and print the FID distance
# fid.update(imgs_true, real=True)
# fid.update(imgs_false, real=False)
# print(fid.compute())


import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from FID_function import FID

print(FID('../split_datasets/mnist3/part_1/', '../samples/demo_mnist_3/'))