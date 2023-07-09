import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def FID(path_real_images, path_generated_images, features = 64):

    fid = FrechetInceptionDistance(feature=features)

    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    imgs_true = ImageFolder(root=path_real_images, transform=transform)
    imgs_false = ImageFolder(root=path_generated_images, transform=transform)

    # Create dataloaders for the datasets
    #dataloader_true = DataLoader(imgs_true, batch_size=32, shuffle=False)
    #dataloader_false = DataLoader(imgs_false, batch_size=32, shuffle=False)

    print(type(imgs_true), len(imgs_true))







    #images, labels = next(iter(imgs_true))
    #print(type(images), images.size(), images)
    #print(type(dataloader_true), type(images))
    #print(len(images))

    #for i, j in enumerate(images):
        #print(i, j)

    # Calculate and print the FID distance
    # fid.update(dataloader_true, real=True)
    # fid.update(dataloader_false, real=False)
    # return fid.compute()

print(FID('../split_datasets/mnist3/', '../samples/demo_mnist_3/'))