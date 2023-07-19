# Want to calculate the expectation of a probabilistic circuit,
# Will start by simply calculating expectation of first pixel which
# should almost cetainly be zero right?

import numpy as np
import torch
import os
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils
import my_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load in the model I have already trained
model_file = os.path.join('../models/einet/demo_mnist_5/', "einet.mdl")
einet = torch.load(model_file)

# # Print information of the model for user
# print("Loaded model from {}".format(model_file))
# print(einet)

# # Define the vector x for feedforward
# #x = torch.from_numpy(np.random.rand(2,784))
# #print(x, type(x))


# # einet.forward(x)

# # Load in mnist
# # train_x, train_labels, test_x, test_labels = datasets.load_mnist()

# # # validation split
# # valid_x = train_x[-10000:, :]
# # train_x = train_x[:-10000, :]
# # valid_labels = train_labels[-10000:]
# # train_labels = train_labels[:-10000]

# # pick the selected classes
# # classes = [3]
# # if classes is not None:
# #     train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
# #     valid_x = valid_x[np.any(np.stack([valid_labels == c for c in classes], 1), 1), :]
# #     test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

# # train_x = torch.from_numpy(train_x).to(torch.device(device))
# # valid_x = torch.from_numpy(valid_x).to(torch.device(device))
# # test_x = torch.from_numpy(test_x).to(torch.device(device))


# # train_N = train_x.shape[0]

# # print(train_N)
# #x = train_x[1:3,:]


# # Forward pass of model with random inputs
# x = torch.randint(0,255, (15,784))/255


# # print(x.shape)
# # #print(x[1])
# # print(einet.forward(x))
# # #print(einet.get_marginalization_idx())

# # dist_layer = einet.einet_layers[0]

# # phi = dist_layer.ef_array.params.squeeze()

# # phi = torch.transpose(phi,0,1)

# # print(f'phi shape is {phi.shape}')

# # # Set all but first row of phi to 1 and then somehow skip the first layer of the forward pass
# # print(phi[:,1])



# # #einet.forward(phi)



# # #marg_idx = (range(1, 784))

# # #einet.set_marginalization_idx(marg_idx)

# # #print(einet.forward(phi))

# # #print(einet.get_marginalization_idx())
# # pixel = 500

# # phi = torch.transpose(phi,0,1)
# # print(phi.shape)
# # #phi[1:,] = 0
# # phi2 = torch.zeros((784,15))
# # phi2[pixel,:] = phi[pixel,:]


# # print(phi2)
# # phi2= phi2.unsqueeze(0)
# # phi2= phi2.unsqueeze(3)
# # # print(phi.shape)

# # print(dist_layer.expectation(phi2))
# # print(einet.expectation(phi2))




# def pixel_expectation(pixel, phi, model):
#     return model.expectation(phi)

# def image_expectation(num_pixels):

#     model_file = os.path.join('../models/einet/demo_mnist_5/', "einet.mdl")
#     einet = torch.load(model_file)
#     dist_layer = einet.einet_layers[0]
#     phi = dist_layer.ef_array.params.squeeze()

#     expectations = torch.zeros(784)

#     print(phi.shape, phi)

#     for pix in range(num_pixels):
#         phi2 = torch.zeros((784,15))
#         phi2[pix,:] = phi[pix,:]
#         phi2 = phi2.unsqueeze(0)
#         phi2 = phi2.unsqueeze(3)
#         expectations[pix] = pixel_expectation(pix,phi2, einet)
        
#     return expectations.detach()

# #print(image_expectation(100))
# image = image_expectation(100)

# exp_dir = '../expectation/demo_mnist_5/1/'
    
# utils.mkdir_p(exp_dir)


# utils.save_image_stack(torch.reshape(image,(1,28,28)), 1, 1, os.path.join(exp_dir, f'exp_image_2.png'), margin_gray_val=0.)

#############################################################################
# Find the Variances of the distribution units

dist_layer = einet.einet_layers[0]

phi = dist_layer.ef_array.params

#print(phi.shape, phi)



def get_variance(phi):

    theta = dist_layer.expectation_to_natural(phi)
    print(theta.shape)

    return variance


get_variance(phi)

def image_expectation(num_pixels, model_file):

    #model_file = os.path.join('../models/einet/demo_mnist_5/', "einet.mdl")
    model = torch.load(model_file)
    dist_layer = model.einet_layers[0]

    phi = dist_layer.ef_array.params.unsqueeze(0)
    
    # If Gaussian
    phi = dist_layer.ef_array.params#.squeeze()
    phi = phi[:,:,:,0]
    print(phi.shape)

    expectations = torch.zeros(784)


    for pix in range(num_pixels):

        phi2 = torch.zeros((1,784,15,1))

        # phi2[:,pix,:,:] = phi[pix,:]

        phi2[:,pix,:,:] = phi[pix,:]
        expectations[pix] = model.expectation(phi2)
        print(pix)
        
    return expectations.detach()