import numpy as np
import my_utils, utils, metrics
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation, ULA
import datasets
import os
from torchmetrics.image import PeakSignalNoiseRatio
import math
import time


# Computing the gradient of the PC prior
def logPC_gradient(prior, x):
    return PC_gradient(prior, x)/prior(x.reshape(1,784,1))

def PC_gradient(prior, x):
    prior(x.reshape(1,784,1)).backward()
    return x.grad

def loglikelihood_gradient(x,y,A,AT,sig2):
    return -torch.reshape(AT@(A@torch.reshape(x,(784,1))-torch.reshape(y, (784,1))),(28,28))/sig2

def MCMC_kernel(x,x2, y, A, AT, sig2, prior, h):
    return x + h * loglikelihood_gradient(x,y,A,AT,sig2) + h * (logPC_gradient(prior,x2)+0.5) + math.sqrt(2*h) * torch.randn_like(x)


psnr = PeakSignalNoiseRatio(data_range = 1)

sig2s = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
classes_list = [[7], None]

# Load the true and noisy images
true_image = my_utils.load_images('../experiments/denoising/Figure/True_images/',grey_scale=True)

# Parameters and initialisations
maxit = 5000
burnin = np.int64(maxit*0.05)
n_samples = np.int64(1000)
psnr_values = []
posterior_means = []
psnrs = []
final_psnrs= []

inpainted = torch.load('../experiments/inpainting/0.5_noisy.pt')
A = torch.diagflat(torch.load('../experiments/inpainting/0.5_A_matrix.pt'))
AT = A


for idx,sig2 in enumerate(sig2s):
    noisy_image = inpainted[idx].reshape(28,28).float()
    print(torch.min(noisy_image), torch.max(noisy_image))

    # Sanity Check
    print(psnr(noisy_image, true_image))

    for classes in classes_list:

        psnr_values = []

        # Initialise the algorithm
        h = 0.01 * sig2
        X = noisy_image.clone()
        X2 = noisy_image.clone() - 0.5
        X2.requires_grad_(True)

        if classes == None:
            i = 'whole'
        else:
            i = classes[0]
        print(i)

        einet = torch.load(f'../experiments/prior/models/{i}_pd_7/einet.mdl')

        start = time.time()

        for j in range(maxit):

            # Update X
            X = MCMC_kernel(X, X2, noisy_image, A, AT, 0.01, einet, h)

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
                # psnr_values.append(psnr(current_mean,true_image))


                if j % 100 == 0:
                    print(j,psnr(current_mean.detach() + 0.5,true_image))

            X = X.detach()
            X2 = (X-0.5).requires_grad_(True)

        print(f'Time taken was {time.time() - start}')

        posterior_means.append(post_meanvar.get_mean().detach())
        psnrs.append(f'mnist_{i}, sig2 = {sig2}: {psnr(current_mean.detach() + 0.5,true_image):.2f}')

    print(psnrs)


utils.mkdir_p('../experiments/inpainting/PC_MCMC')
torch.save(posterior_means,'../experiments/inpainting/PC_MCMC/final_inpainting_results.pt')





