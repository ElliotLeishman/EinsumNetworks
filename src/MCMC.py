import numpy as np
import my_utils, utils, metrics, ULA
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os
from torchmetrics.image import PeakSignalNoiseRatio
import time as time
import tqdm, math

# Computing the gradient of the PC prior
def logPC_gradient(model, x):
    model(x.reshape(1,784,1)).backward()    
    return x.grad/model(x.reshape(1,784,1))

# ULA kernel update
def ULA_kernel(X, delta):
    return X - delta * gradlogpi(X) + math.sqrt(2*delta) * torch.randn_like(X)

sig2 = 0.05

# Load image
x = my_utils.load_images('../experiments/prior/mnist_[7]/test', grey_scale = True)[:100,:].squeeze()

# Apply the blur operator
A,AT,AAT_norm = forward_models.blurring(x, save_dir = '../blur/test/2/', sig2 = sig2)

# Define the likelihood functions
f = lambda x,A : (torch.linalg.matrix_norm(y-A(x), ord=2)**2.0)/(2.0*sig2)
gradf = lambda x,A,AT : AT(A(x)-y)/sig2
L_y = AAT_norm/(sig2)

# Prior (PC)
einet = torch.load('../experiments/prior/models/7_pd_7/einet.mdl')
PC = lambda x : einet(x)

# Posterior and gradient
logpi = lambda z: (-f(z,A) - torch.log(PC(z.reshape(1,784,1))))
gradlogpi = lambda x: gradf(x,A,AT) + (logPC_gradient(einet, x))

# Stepsize
delta = 0.5

# Load a blurred image and reshape it to suitable size
y = my_utils.load_images('../blur/test/', grey_scale = True)[0,:]

# Algorithm initialisation
maxit = 100
#burnin = np.int64(maxit*0.05)
burnin =2
n_samples = np.int64(maxit)
X = y.clone().requires_grad_(True)
#X = torch.rand((28,28), requires_grad = True)
MC_X = []
thinned_trace_counter = 0
thinning_step = np.int64(maxit/n_samples)
nrmse_values = []
psnr_values = []
ssim_values = []
log_pi_trace = []

# Algorithm
start_time = time.time()
for i_x in range(maxit):

    # Update X
    X = ULA_kernel(X, delta)
    if i_x == burnin:
        # Initialise recording of sample summary statistics after burnin period
        post_meanvar = ULA.welford(X)
        absfouriercoeff = ULA.welford(torch.fft.fft2(X).abs())
        count=0
    elif i_x > burnin:
        # update the sample summary statistics
        post_meanvar.update(X)
        absfouriercoeff.update(torch.fft.fft2(X).abs())

        # collect quality measurements
        current_mean = post_meanvar.get_mean()
        #nrmse_values.append(NRMSE(x, current_mean))
        #psnr_values.append(PSNR(x, current_mean))
        #ssim_values.append(SSIM(x, current_mean))
        #log_pi_trace.append(logpi(X).detach().item())

        # collect thinned trace
        if count == thinning_step-1:
            MC_X.append(X.detach().cpu().numpy())
            count = 0
        else:
            count += 1

    X = X.detach().requires_grad_(True)
    print(i_x)

end_time = time.time()
elapsed = end_time - start_time 
print(elapsed)      

# Save the stuff for plots
np.save(f'../blur/means.npy', post_meanvar.get_mean().detach().cpu().numpy())
