# Test to see if PyTorch will give me grad(log(p(x|y)))
# There must be a way to do this - I just don't know it rn

import my_utils, utils, metrics
import torch
from EinsumNetwork import Graph, EinsumNetwork
import forward_models, expectation
import datasets
import os


# Computing the gradient of the PC prior
def logPC_gradient(model, x):
    model(x.reshape(1,784,1)).backward()    
    return x.grad/model(x.reshape(1,784,1))


# Load in the model
einet = torch.load('../models/einet/demo_mnist_7/einet.mdl')
print(einet)

# Initialise something for gradient to be evaluated at
x = torch.rand((28,28), requires_grad = True)


loggrad = logPC_gradient(einet, x)
print(loggrad,loggrad.shape)

loggrad = loggrad.detach().requires_grad_(True)


print(logPC_gradient(einet,loggrad))










# y = einet(x.reshape((1,784,1)))
# print(y,y.shape)


# # Perform backpropagation to compute gradients with respect to x
# y.backward()

# # Access the gradient with respect to x
# gradient = x.grad


# print(gradient)
# #print(gradient.shape)