import my_utils, utils, metrics
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os

sig2s = [0.00001]

# FIGURE
#############################################################################
#Load in the clean images

clean_images = my_utils.load_images('../experiments/denoising/Figure/True_images', grey_scale=True)

# Restore the images
for sig in sig2s:

    # Apply forward model
    forward_models.inpainting(clean_images, 0.5, noise = True, sigma = sig, save = True, save_dir = f'../experiments/inpainting/Figure/Noisy_images/sigma_{sig}/1')

    # Load images
    missing_images = my_utils.load_images(f'../experiments/inpainting/Figure/Noisy_images/', grey_scale = True)

    # Make directory to save images
    save_dir = f'../experiments/inpainting/Figure/Restored_images/sigma_{sig}/1'
    utils.mkdir_p(save_dir)

    for i in range(2):
        expectation.depainting('../models/einet/demo_mnist/einet.mdl', missing_images[0,:], missing_images[i+1,:], K=10, sigma = sig, img_name = f'{i+7}_{sig}', save_dir = save_dir)


#############################################################################

# for sig2 in sig2s:

#     # Create the noisy images using test mnist and save them
#     true_images = my_utils.load_images('../mnist/whole/test/', grey_scale = True)
#     noisy_images, A = forward_models.inpainting(true_images, prop=0.5, noise=True, sigma = sig2, save = True, save_dir = f'../experiments/inpainting/noisy/sigma_{sig2}/1')
#     noisy_images = noisy_images[:1000,:]

#     # Make directory to save images
#     save_dir = f'../experiments/inpainting/denoisy/sigma_{sig2}/1'
#     utils.mkdir_p(save_dir)

#     # Denoise the noisy images
#     for i in range(1, noisy_images.shape[0]):
#         denoisy_images = expectation.depainting('../models/einet/demo_mnist/einet.mdl',  A, noisy_images[i], K=10, sigma = sig2, img_name = f'image_{i}', save_dir = save_dir)    
#         utils.save_image_stack(torch.reshape(denoisy_images,(1,28,28)), 1, 1, os.path.join(save_dir, f'image_{i}.png'))
        
#         if i%10 == 0:
#             print(f'Image {i} for variance {sig2} denoised.')

# # DIFFERENT PROPORTIONS OF A
# ######################################################################
# # sig2 = 0.0001
# # props = [0.1,0.25,0.33,0.5,0.75]

# # # Load Clean images
# # clean_images = my_utils.load_images('../experiments/denoising/Figure/True_images', grey_scale=True)

# # for prop in props:

# #     # Apply forward model
#     noisy_images, A = forward_models.inpainting(clean_images, prop=prop, noise=True, sigma = sig2, save = True, save_dir = f'../experiments/inpainting_prop/Figure/Noisy_images/prop_{prop}/1')

#     # Make the directory
#     save_dir = f'../experiments/inpainting_prop/Figure/restored/prop_{prop}/1'
#     utils.mkdir_p(save_dir)

#     # Restore images
#     for i in range(2):
#         expectation.depainting('../models/einet/demo_mnist/einet.mdl', A, noisy_images[i,:], K=10, sigma = sig2, img_name = f'{i+7}_{prop}', save_dir = save_dir)






