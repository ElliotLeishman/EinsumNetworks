# Script which will train a better model and then return an image
# of the expectation
import numpy as np
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import utils
import datasets
from EinsumNetwork import Graph, EinsumNetwork
import time
import my_utils

my_utils.train_model('../models/',[7], EinsumNetork.NormalArray, K = 15, pd_pieces = [15], fashion_mnist = False, num_epochs = 20, batch_size = 100)

my_utils.train_model('../models/all/',None, EinsumNetork.NormalArray, K = 15, pd_pieces = [10], fashion_mnist = False, num_epochs = 15, batch_size = 100)
