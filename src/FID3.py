import torch
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader



fid = FrechetInceptionDistance(feature=64)


# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



imgs_dist1 = ImageFolder(root='../split_datasets/mnist3/part_1/', transform=transform)
imgs_dist2 = ImageFolder(root='../samples/demo_mnist_3/', transform=transform)













# generate two slightly overlapping image intensity distributions
#imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
#imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
fid.update(imgs_dist1, real=True)
fid.update(imgs_dist2, real=False)
print(fid.compute())