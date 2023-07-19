import torch
import my_utils
import utils
import os
import numpy as np

image = my_utils.load_images('../blur/', True)[0:50,:]
#image = torch.ones(10,1,28,28)
#image[:,:,0,0] = 0

#print(image)

def create_inpainting_matrix(num_pixels, prop):

    vector = torch.ones(num_pixels)
    vector[torch.randperm(num_pixels)[:int(num_pixels * prop)]] = 0

    return vector



def inpainting(image, prop, noise = True, sigma = 0.05, save = False, save_dir = None):

    shape = image.shape
    d = shape[2] * shape[3]
    image = torch.reshape(image, (shape[0], shape[1], d))

    A = create_inpainting_matrix(image.shape[2], prop)

    #image =  A * image
    if noise:
        gauss_noise = np.random.multivariate_normal(np.zeros(d), sigma * np.eye(d), shape[0])
        image = A * image + torch.from_numpy(gauss_noise).unsqueeze(1)
    else:
        image = A * image

    if save and save_dir is not None:
        utils.mkdir_p(save_dir)
        utils.save_image_stack(torch.reshape(A,(1,shape[2],shape[3])), 1, 1, os.path.join(save_dir, f'a_matrix.png'), margin_gray_val=0.)
        for i in range(shape[0]):
            utils.save_image_stack(torch.reshape(image[i],(1,shape[2],shape[3])), 1, 1, os.path.join(save_dir, f'inpainted_{i}.png'), margin_gray_val=0.)
        print(f'Images saved to {save_dir}')
    elif save:
        print('Need name of directory to save images to...')

    return image, A

#inpainting(image, 0.5, save = True, save_dir = '../inpainting/1')


def inverse_inpainting(model_file, noisy_image, epsilon, A, num_pixels, batch_size, K = 15, gaussian = True, save = False, save_dir = None):

    # Get model
    einet = torch.load(model_file)
    dist_layer = einet.einet_layers[0]

    # Format noisy image
    y = torch.reshape(noisy_image,(1,784)).squeeze()
    y = torch.transpose(y.repeat(K,1),0,1).unsqueeze(2)

    # Distibution parameters
    phi = dist_layer.ef_array.params
    mean, var = gaussian_product(phi, y, epsilon, dist_layer)

noisy_image = my_utils.load_images('../inpainting/')[0:,]

my_utils.denoising_expectation('../models/einet/demo_mnist_5/',noisy_image, 0.05, 784, 28, K = 10, save = True, save_dir = f'../depainted/5/')