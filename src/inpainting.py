import numpy as np
import my_utils, utils, metrics, ULA
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os
from torchmetrics.image import PeakSignalNoiseRatio
import time as time

# Code that will get me results for multiple denosing instances
# Should store average psnr for both the noisy and denoisy images
# Using k = 15, pd_pieces = 7

sig2s = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
classes_list = [[7], None]
num_sam = 1000
psnr = PeakSignalNoiseRatio(data_range = 1, reduction = 'none', dim = [1,2,3])

# Load the A matrix
A = torch.load('../experiments/inpainting/0.5_A_matrix.pt')
print('Matrix A has been loaded in.')

# Load the models
def inpaint_depaint(classes_list, sig2s):

    for classes in classes_list:
        
        if classes == None:
            i = 'whole'
        else:
            i = classes[0]
        print(i)

        # Intialise the lists to store the summary statistics
        means = []
        sds = []
        means2 = []
        sds2 = []

        # Load the true images
        #true_images = my_utils.load_images(f'../experiments/prior/mnist_{i}/test/',grey_scale=True)[:num_sam,:]
        #true_images = torch.reshape(true_images, (-1,1,28,28))

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
        utils.mkdir_p(f'../experiments/prior/mnist_{i}/train/1')
        utils.mkdir_p(f'../experiments/prior/mnist_{i}/test/1/')

        # # Save the split up MNIST
        # for j in range(train_x.shape[0]):
        #     utils.save_image_stack(torch.reshape(torch.from_numpy(train_x[j,:]),(1,28,28)), 1, 1, filename = f'../experiments/prior/mnist_{i}/train/1/image_{j}.png', margin_gray_val=0.)

        # for j in range(test_x.shape[0]):
        #     utils.save_image_stack(torch.reshape(torch.from_numpy(test_x[j,:]),(1,28,28)), 1, 1, filename = f'../experiments/prior/mnist_{i}/test/1/image_{j}.png', margin_gray_val=0.)

        # # Load the relevant mnist datasets
        # test_images = my_utils.load_images(f'../experiments/prior/mnist_{i}/test/')

        # Load the PC:
        model = torch.load(f'../experiments/prior/models/{i}_pd_7/einet.mdl')

        test_x = torch.from_numpy(test_x).reshape(-1,1,28,28)
        train_x = torch.from_numpy(train_x).reshape(-1,1,28,28)


        for sig2 in sig2s:

            noisy_images = torch.zeros((num_sam,784))
            As = torch.zeros((num_sam,784))

            for im in range(num_sam):
                # Create the noisy images
                noisy_images[im,:], As[im,:] = forward_models.inpainting(test_x[im,:].reshape(1,1,28,28), 0.5, sigma = sig2)


            print(f'Inpainted images have been created. The shape is {noisy_images.shape}')

            noisy_images = torch.reshape(noisy_images, (-1,1,28,28))

            # Remove the noise
            denoised_images = torch.zeros((num_sam,784))
            for j in range(num_sam):
                denoised_images[j,:] = expectation.depainting(model, As[j], noisy_images[j], K = 15, sigma = sig2)

            denoised_images = torch.reshape(denoised_images, (-1,1,28,28))
            print(f'Inpainted images have been restored. The shape is {denoised_images.shape}')


            # Update Summary statistics
            psnr1 = psnr(denoised_images[:num_sam,:] + 0.5,test_x[:num_sam,:] + 0.5)
            psnr2 = psnr(noisy_images[:num_sam,:] + 0.5,test_x[:num_sam,:] + 0.5)

            means.append(f'sig2 = {sig2}: {torch.mean(psnr1):.2f}')
            sds.append(f'sig2 = {sig2}: {torch.std(psnr1):.2f}')
            means2.append(f'sig2 = {sig2}: {torch.mean(psnr2):.2f}')
            sds2.append(f'sig2 = {sig2}: {torch.std(psnr2):.2f}')

            print(means, sds, means2, sds2)

            # Create the directories to save the images
            utils.mkdir_p(f'../experiments/inpainting50/painted/')
            utils.mkdir_p(f'../experiments/inpainting50/depainted/')

            # Save the noisy and denoised images as raw tensors
            torch.save(noisy_images[:num_sam,:], f'../experiments/inpainting50/painted/mnist_{i}_sigma_{sig2}.pt')
            torch.save(denoised_images[:num_sam,:], f'../experiments/inpainting50/depainted/mnist_{i}_sigma_{sig2}.pt')


        print(f'Summary statistics for PSNR Values, (denoised & true) (mnist_{i}) means: {means}, sds: {sds}')
        print(f'Summary statistics for PSNR Values, (noisy & true) (mnist_{i}) means: {means2}, sds: {sds2}')




inpaint_depaint(classes_list, sig2s)