import my_utils, utils, metrics
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os
from torchmetrics.image import PeakSignalNoiseRatio

# Params
num_sam = 1000
batch_size = 100

for classes in [None]:

    if classes == None:
        i = 'whole'
    else:
        i = classes[0]

    # Split MNIST dataset up
    ####################################################
    train_x, train_labels, test_x, test_labels = datasets.load_mnist()
    train_x /= 255.
    test_x /= 255.
    train_x -= .5
    test_x -= .5

    # Select the right class
    if classes is not None:
        train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
        test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

    test_x = torch.tensor(test_x[:num_sam,:])
    test_x = 75 * test_x + 127.5
    test_x = test_x.to(torch.uint8)
    test_x = test_x.reshape((-1,1,28,28)).repeat(1,3,1,1)

    train_x = torch.tensor(train_x[:num_sam,:])
    train_x = 75 * train_x + 127.5
    train_x = train_x.to(torch.uint8)
    train_x = train_x.reshape((-1,1,28,28)).repeat(1,3,1,1)


    for pd in [2,4,7]:

        # Load the model
        prior = torch.load(f'../experiments/prior/models/{i}_pd_{pd}/einet.mdl')

        # Generate the samples
        samples = prior.sample(num_samples = num_sam)
        samples = 75 * samples + 127.5
        print(torch.min(samples), torch.max(samples))
        samples = samples.to(torch.uint8)
        samples = samples.reshape((-1,1,28,28)).repeat(1,3,1,1)

        FID_values = metrics.FID_function(test_x, samples, batch_size, 2048)
        print(f'The FID Values for {i}_pd_{pd} are: {FID_values}')

    print(f'For class {i} FID values comparing test and train are: {metrics.FID_function(test_x, train_x, batch_size, 2048)}')


