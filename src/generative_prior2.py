import numpy as np
import my_utils, utils, metrics
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os
from torchmetrics.image import PeakSignalNoiseRatio
import time as time


num_epochs = 10 # change this to 10
K = 15
num_sam = 500 # maybe change this as well?

# Initialise the vectors
timing = []
values = []


# want to test the effect of number of pd_pieces on time taken and sample quality
for classes in [[7], None]:

    if classes == None:
        i = 'whole'
    else:
        i = classes[0]
    print(i)

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

    # Make directory
    utils.mkdir_p(f'../experiments/prior/mnist_{i}/train/1/')
    utils.mkdir_p(f'../experiments/prior/mnist_{i}/test/1/')

    # Save the split up MNIST
    for j in range(train_x.shape[0]):
        utils.save_image_stack(torch.reshape(torch.from_numpy(train_x[j,:]),(1,28,28)), 1, 1, filename = f'../experiments/prior/mnist_{i}/train/1/image_{j}.png', margin_gray_val=0.)

    for j in range(test_x.shape[0]):
        utils.save_image_stack(torch.reshape(torch.from_numpy(test_x[j,:]),(1,28,28)), 1, 1, filename = f'../experiments/prior/mnist_{i}/test/1/image_{j}.png', margin_gray_val=0.)

    true = my_utils.load_images(f'../experiments/prior/mnist_{i}/test/', grey_scale = False).type(torch.uint8)[:num_sam,:]

    for pd in [2,4,7]:
    
        starting = time.time()
        # Train up the model
        # batch_size = 100, online_em_frequency = 1, online_em_stepsize = 0.05
        my_utils.train_model(f'../experiments/prior/models/{i}_pd_{pd}/', classes, EinsumNetwork.NormalArray, K = 15, pd_pieces = [pd], fashion_mnist = False, num_epochs = num_epochs)
        timing.append(f'Total training time for {i}_pd_{pd} is: {time.time()-starting}')

        # Create num_sam samples from each prior
        my_utils.sampling(num_sam,f'../experiments/prior/models/{i}_pd_{pd}/', f'../experiments/prior/samples/{i}_pd_{pd}/1/', True)

        # Save the expectation of each prior
        utils.mkdir_p(f'../experiments/prior/expectation/')
        expectation.image_expectation(torch.load(f'../experiments/prior/models/{i}_pd_{pd}/einet.mdl'), 784, 28, K = 15, gaussian = True, means = None, save = True, save_dir = f'../experiments/prior/expectation/{i}_pd_{pd}.png')

        print(timing)

        # FID Part
        ######################################################################
        # Reload the test data

        gen = my_utils.load_images(f'../experiments/prior/samples/{i}_pd_{pd}/', grey_scale = False).type(torch.uint8)[:num_sam,:]
        values.append(f'The FID values for {i}_pd_{pd} are:{metrics.FID_function(true,gen,100,2048)}')

    train = my_utils.load_images(f'../experiments/prior/mnist_{i}/train/', grey_scale = False).type(torch.uint8)[:num_sam,:]
    values.append(f'The FID values for {i} train vs. test are:{metrics.FID_function(true,train,100,2048)}')



print(timing)
print(values)