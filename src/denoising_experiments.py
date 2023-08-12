import my_utils, utils, metrics
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os
from torchmetrics.image import PeakSignalNoiseRatio


# Take a PC trained on a number of whole of MNIST and then noise up some test images with varying degrees of noise.
# Calculate the PSNR of the images compare to noisy images and then calculate denoise the images
# Then calculate PSNR again and it should be lower.
# Repeat a bunch of times for reliability.
# Make some pretty pictures.


# Load the model
#einet = torch.load('../models/einet/demo_mnist/einet.mdl')
#print(einet)
#sigma = 0.01

# Save Mnist - Can change to just specific numbers pretty easily i think 
# train_x, train_labels, test_x, test_labels = datasets.load_mnist()
# train_x = torch.reshape(torch.from_numpy(train_x).unsqueeze(1),(-1,1,28,28))
# test_x = torch.reshape(torch.from_numpy(test_x).unsqueeze(1),(-1,1,28,28))


# train_set ='../mnist/whole/train/1'
# utils.mkdir_p(train_set)
# for i in range(train_x.shape[0]):
#     utils.save_image_stack(train_x[i], 1, 1, os.path.join(train_set, f'image_{i}.png'), margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0)


# test_set ='../mnist/whole/test/1'
# utils.mkdir_p(test_set)
# for i in range(test_x.shape[0]):
#     utils.save_image_stack(test_x[i], 1, 1, os.path.join(test_set, f'image_{i}.png'), margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0)

# Noise up the images
# noisy_images = forward_models.gaussian_noise('../mnist/whole/train/', sigma = sigma, save = True, save_dir = f'../experiments/denoising/noisy/sigma_{sigma}')
# Load noisy images
#noisy_images = forward_models.gaussian_noise('../experiments/test/', sigma = sigma)

#denoised = f'../experiments/denoising/denoised/sigma_{sigma}'
#utils.mkdir_p(denoised)

# Denoise the images - Note that the functions currently only work with one image at a time
#for i in range(noisy_images.shape[0]):
#    denoisy_images = expectation.denoising_expectation('../models/einet/demo_mnist/einet.mdl', noisy_images[i], sigma, 784, 28, K = 10, gaussian = True, save = False, save_dir = None)
    
    
#    utils.save_image_stack(torch.reshape(denoisy_images,(1,28,28)), 1, 1, os.path.join(denoised, f'image_{i}.png'), margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0)


# Noise the images
#utils.mkdir_p('../experiments/denoising/Figure/True_images/1')

######### FIGURE CODE
sig2s = [0.000001, 0.0001, 0.01]
for sig in sig2s:
    noisy_images = forward_models.gaussian_noise('../experiments/denoising/Figure/True_images', sigma = sig)
    noisy_images = forward_models.gaussian_noise('../experiments/denoising/Figure/True_images', sigma = sig, save = True, save_dir = f'../experiments/denoising/Figure/Noisy_images/sigma_{sig}/1')

# Denoise the images
for sig in sig2s:
    
    # Load images
    noise_images = my_utils.load_images(f'../experiments/denoising/Figure/Noisy_images/sigma_{sig}', grey_scale = True)

    # Make directory to save images
    save_dir = f'../experiments/denoising/Figure/Denoisy_images/sigma_{sig}/1'
    utils.mkdir_p(save_dir)

    for i in range(2):
        expectation.denoising_expectation('../models/einet/demo_mnist/einet.mdl', noise_images[i,:], sig, 784, 28, K = 10, gaussian = True, save = True, save_dir = os.path.join(save_dir,f'{i+7}_denoised_{sig}.png'))

#sig2s = [0.01, 0.05, 0.1, 0.25,1]
#sig2s = [0.01, 0.05]#, 0.1, 0.25,1]


############################################################################################################################

# for sig2 in sig# 2s:

#     Create the noisy images using test mnist and save them 
#     noisy_images = forward_models.gaussian_noise('../mnist/whole/test/', sigma = sig2, save = True, save_dir = f'../experiments/denoising/noisy/sigma_{sig2}/1')[:1000,:]

#     Make directory to save images
#     save_dir = f'../experiments/denoising/noisy/sigma_{sig2}/1'
#     utils.mkdir_p(save_dir)

#     # Denoise the noisy images
#     for i in range(noisy_images.shape[0]):
#         denoisy_images = expectation.denoising_expectation('../models/einet/demo_mnist/einet.mdl', noisy_images[i], sig2, 784, 56, K = 10, gaussian = True, save = False, save_dir = None)    
#         utils.save_image_stack(torch.reshape(denoisy_images,(1,28,28)), 1, 1, os.path.join(save_dir, f'image_{i}.png'))
        
#         if i%10 == 0:
#             print(f'Image {i} for variance {sig2} denoised.')


##############################################################################################################################

# Summary statistics for denoising
def summary_statistics(sig2s):

    # Intialise the lists to store the summary statistics
    means = []
    sds = []
    means2 = []
    sds2 = []

    # Load the true images
    true_images = my_utils.load_images('../mnist/whole/test/',grey_scale=True)[:1000,:]

    for sig2 in sig2s:

        # Load the denoised images
        denoised_images = my_utils.load_images(f'../experiments/denoising/Denoised_images/sigma_{sig2}/', grey_scale=True)

        #print(true_images.shape, denoised_images.shape)

        psnr = PeakSignalNoiseRatio(data_range = 1, reduction = 'none', dim = [1,2,3])

        psnrs = psnr(denoised_images,true_images)

        #print(psnrs)


        # Update the summary statistics
        means.append(torch.mean(psnrs))
        sds.append(torch.std(psnrs))

    for sig2 in sig2s:

        # True images
        noisy_images = my_utils.load_images(f'../experiments/denoising/noisy/sigma_{sig2}/',grey_scale=True)[:1000,:]

        # Load the denoised images
        denoised_images = my_utils.load_images(f'../experiments/denoising/Denoised_images/sigma_{sig2}/', grey_scale=True)

        #print(true_images.shape, denoised_images.shape)

        psnr = PeakSignalNoiseRatio(data_range = 1, reduction = 'none', dim = [1,2,3])

        psnrs = psnr(noisy_images,true_images)

        #print(psnrs)


        # Update the summary statistics
        means2.append(torch.mean(psnrs))
        sds2.append(torch.std(psnrs))    




    print(means,sds)
    print(means2,sds2)



#summary_statistics(sig2s)



# Check that PSNR Values agree with those from sampling_tutorials
# denoised_images = my_utils.load_images('../experiments/denoising/Figure/Denoisy_images/',grey_scale=True)[[0,2,4,6,8],:]

# noisy_images = my_utils.load_images(f'../experiments/denoising/Figure/Noisy_images',grey_scale=True)[[0,2,4,6,8],:]

# psnr = PeakSignalNoiseRatio(data_range = 1, reduction = 'none', dim = [1,2,3])

# print(psnr(noisy_images,denoised_images))
# print(psnr(denoised_images,noisy_images))





