import numpy as np
import my_utils, utils, metrics
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation, ULA
import datasets
import os
from torchmetrics.image import PeakSignalNoiseRatio
import math


# Computing the gradient of the PC prior
def logPC_gradient(prior, x):
    return PC_gradient(prior, x)/prior(x.reshape(1,784,1))

def PC_gradient(prior, x):
    prior(x.reshape(1,784,1)).backward()
    return x.grad

def logposterior_gradient(x,y,sig2):
    return -(x-y)/sig2

def MCMC_kernel(x, x2, y, sig2, prior, h):
    return x + h * logposterior_gradient(x,y,sig2) + h * (logPC_gradient(prior,x2)+ 0.5) + math.sqrt(2*h) * torch.randn_like(x)

i = 7
psnr = PeakSignalNoiseRatio(data_range = 1)

sig2s = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
classes_list = [[7], None]

# Load the true and noisy images
true_image = my_utils.load_images('../experiments/denoising/Figure/True_images/',grey_scale=True)

# Parameters and initialisations
maxit = 25000
burnin = np.int64(maxit*0.05)
n_samples = np.int64(1000)
psnr_values = []
posterior_means = []
psnrs = []
final_psnrs = []

for sig2 in sig2s:
    noisy_image = torch.load(f'../experiments/denoising2/Figure/Noisy_images/7_sig{sig2}.pt').reshape((28,28)).type(torch.float32)

    # Sanity Check
    print(psnr(noisy_image, true_image))

    for classes in classes_list:

        psnr_values = []

        # Initialise the algorithm
        h = sig2
        X = noisy_image.clone()
        X2 = noisy_image.clone() - 0.5
        X2.requires_grad_(True)

        if classes == None:
            i = 'whole'
        else:
            i = classes[0]
        print(i)

        einet = torch.load(f'../experiments/prior/models/{i}_pd_7/einet.mdl')



        for j in range(maxit):

            # Update X
            X = MCMC_kernel(X, X2, noisy_image, 0.01, einet, h)

            if j == burnin:
                # Initialise recording of sample summary statistics after burnin period
                post_meanvar = ULA.welford(X)
                absfouriercoeff = ULA.welford(torch.fft.fft2(X).abs())

            elif j > burnin:
                # update the sample summary statistics
                post_meanvar.update(X)
                absfouriercoeff.update(torch.fft.fft2(X).abs())

                # collect quality measurements
                current_mean = post_meanvar.get_mean()

                if j % 250 == 0:
                    print(j,psnr(current_mean.detach(),true_image))

            X = X.detach()
            X2 = (X-0.5).requires_grad_(True)


        posterior_means.append(post_meanvar.get_mean().detach())
        final_psnrs.append(f'mnist_{i}, sig2 = {sig2}: {psnr(current_mean.detach(),true_image):.2f}')
        torch.save(current_mean.detach(), f'../experiments/denoising2/PC_MCMC/{i}_{sig2}.pt')
    print(final_psnrs)


utils.mkdir_p('../experiments/denoising2/PC_MCMC')
torch.save(posterior_means,'../experiments/denoising2/PC_MCMC/post_means_2.pt')
torch.save(psnrs, '../experiments/denoising2/PC_MCMC/psnr_values')
torch.save(final_psnrs, '../experiments/denoising2/PC_MCMC/final_psnr_values')




