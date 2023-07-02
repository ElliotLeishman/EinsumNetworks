import torch
import torchvision
import torchmetrics
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
#from ignite.metrics import * - not working

print(torchvision.__version__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the datasets
part1_dataset = ImageFolder(root='../split_datasets/mnist/', transform=transform)
part2_dataset = ImageFolder(root='../split_datasets/mnist/', transform=transform)

print(part1_dataset)





metric = FID()
metric.attach(default_evaluator, "fid")
y_true = torch.ones(10, 4)
y_pred = torch.ones(10, 4)
state = default_evaluator.run([[y_pred, y_true]])
print(state.metrics["fid"])