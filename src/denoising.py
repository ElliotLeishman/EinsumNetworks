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

#sig2s = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
sig2s = [0.0001]

classes_list = [[7]]
num_sam = 5

# Load the models
def noise_denoise(classes_list, sig2s):

    for classes in classes_list:
        
        print(classes)
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



        # Using k = 15, pd_pieces = 7
        # model = torch.load(f'../experiments/prior/models/{i}_pd_7/einet.mdl')

        # Load the relevant mnist datasets
        test_images = my_utils.load_images(f'../experiments/prior/mnist_{i}/test/')

        for sig2 in sig2s:
            # Create the noisy images
            noisy_images = forward_models.gaussian_noise(f'../experiments/prior/mnist_{i}/test/', sigma = sig2, save = True, save_dir = f'../experiments/denoising2/noisy/mnist_{i}/sigma_{sig2}/1')

            # Create save directory
            denoised = f'../experiments/denoising2/denoised/mnist_{i}/sigma_{sig2}/1'
            utils.mkdir_p(denoised)

            # Remove the noise
            for j in range(num_sam):
                denoisy_images = expectation.denoising_expectation(f'../experiments/prior/models/{i}_pd_7/einet.mdl', noisy_images[j], sig2, 784, 28, K = 15, gaussian = True, save = False, save_dir = None)
        
        
                utils.save_image_stack(torch.reshape(denoisy_images,(1,28,28)), 1, 1, os.path.join(denoised, f'image_{j}.png'), margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0)

def summary_statistics(classes_list, sig2s):

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
        true_images = my_utils.load_images(f'../experiments/prior/mnist_{i}/test/',grey_scale=True)[:num_sam,:]

        for sig2 in sig2s:

            # Load the denoised images
            denoised_images = my_utils.load_images(f'../experiments/denoising2/denoised/mnist_{i}/sigma_{sig2}', grey_scale=True)
            psnr = PeakSignalNoiseRatio(data_range = 1, reduction = 'none', dim = [1,2,3])
            psnrs = psnr(denoised_images,true_images)

            # Update the summary statistics
            means.append(torch.mean(psnrs))
            sds.append(torch.std(psnrs))

            # True images
            noisy_images = my_utils.load_images(f'../experiments/denoising2/noisy/mnist_{i}/sigma_{sig2}',grey_scale=True)[:num_sam,:]

            psnr = PeakSignalNoiseRatio(data_range = 1, reduction = 'none', dim = [1,2,3])
            psnrs = psnr(noisy_images,true_images)

            # Update the summary statistics
            means2.append(torch.mean(psnrs))
            sds2.append(torch.std(psnrs))    




        print(f'Summary statistics for PSNR Values, (denoised & true) means: {means}, sts: {sds}')
        print(f'Summary statistics for PSNR Values, (noisy & true) means: {means2}, sts: {sds2}')

noise_denoise(classes_list, sig2s)
summary_statistics(classes_list, sig2s)







