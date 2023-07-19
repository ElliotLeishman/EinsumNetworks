# Calculate the SNR of an image and then a bunch of images.

import my_utils
import torch

image1 = my_utils.load_images('../blur/', True)[0:100,:,:,:]
image2 = my_utils.load_images('../blur2/', True)[0:1,:,:,:]
image3 = my_utils.load_images('../expectation/demo_mnist_5/', True)[0:100,:]
print(image1.shape, image2.shape)

#print(image.shape)

#print(torch.mean(image, dim = (2,3)))
#print(torch.std(image, dim = (2,3)))

#print(torch.mean(image, dim = (2,3)) / torch.std(image, dim = (2,3)))



def MSE(clean, noisy):
    return torch.sum(torch.square(clean - noisy), dim = (2,3))/784

def PSNR(clean,noisy):
    return -10 * torch.log10(MSE(clean, noisy))

blurred = PSNR(image1,image2)
deblurred = PSNR(image1,image3)

difference = (deblurred - blurred).squeeze()
print(difference)
print(torch.max(difference), torch.min(difference))