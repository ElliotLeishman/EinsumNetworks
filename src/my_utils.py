# Should get rid of these eventually
import numpy as np
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import utils
import datasets
from EinsumNetwork import Graph, EinsumNetwork
import time

def load_images(dir, grey_scale = False):
    '''
    Takes a directory containing images and returns a tensor of those images
    '''

    # Define the transformations
    if grey_scale:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    # Define the image folder and data loader
    img_folder= ImageFolder(dir, transform = transform)
    img_loader = DataLoader(img_folder, batch_size=None)

    images = torch.zeros(len(img_loader), *img_folder[0][0].shape)

    for idx, samples in enumerate(img_folder):
        images[idx] = samples[0]

    return images

def sampling(num_sam, model_dir, sample_dir = None, save = False):

    # Load the model
    model_file = os.path.join(model_dir, "einet.mdl")
    einet = torch.load(model_file)

    # Print information of the model for user
    print("Loaded model from {}".format(model_file))
    print(einet)

    # Sample from the model
    samples = einet.sample(num_samples = num_sam).cpu()

    if save and sample_dir is not None:
        utils.mkdir_p(sample_dir)
        for i in range(num_sam):
            utils.save_image_stack(torch.reshape(samples[i],(1,28,28)),\
             1, 1, os.path.join(sample_dir, f'image_{i}.png'),\
             margin_gray_val=0.)
        print(f'Images saved to {sample_dir}')
    elif save:
        print('Need name of directory to save images to...')

    return samples

def split_mnist(classes, fashion = False, save = False, save_dir = None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the data
    if fashion:
        train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
    else:
        train_x, train_labels, test_x, test_labels = datasets.load_mnist()

    # Pick the selected classes
    if classes is not None:
        train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
        test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

    # Convert to pytorch tensors
    train_x = torch.from_numpy(train_x).to(torch.device(device))
    test_x = torch.from_numpy(test_x).to(torch.device(device))

    # Split the training set into 2
    split = len(train_x) // 2
    train_x1 = train_x[:split]
    train_x2 = train_x[split:]

    if save and save_dir is not None:

        # Set & make directory
        part_1 = os.path.join(save_dir, 'part_1/1')
        part_2 = os.path.join(save_dir, 'part_2/1')

        utils.mkdir_p(part_1)
        utils.mkdir_p(part_2)

        for i in range(len(train_x1)):
            utils.save_image_stack(torch.reshape(train_x1[i],(1,28,28)), 1, 1,\
             os.path.join(part_1, f'image_{i}.png'), margin_gray_val=0.)
            utils.save_image_stack(torch.reshape(train_x2[i],(1,28,28)), 1, 1, \
            os.path.join(part_2, f'image_{i}.png'), margin_gray_val=0.)
        print(f'Images saved to {save_dir}')

    elif save:
        print('Need name of directory to save images to...')

    return train_x1, train_x2


def train_model(save_dir, classes, exp_family, K, pd_pieces, fashion_mnist = False, num_epochs = 1, batch_size = 10, online_em_frequency = 1, online_em_stepsize = 0.05):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Exponential Families
    exponential_family_args = None
    if exp_family == EinsumNetwork.BinomialArray:
        exponential_family_args = {'N': 255}
    if exp_family == EinsumNetwork.CategoricalArray:
        exponential_family_args = {'K': 256}
    if exp_family == EinsumNetwork.NormalArray:
        exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

    # Load in the data
    if fashion_mnist:
        train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
    else:
        train_x, train_labels, test_x, test_labels = datasets.load_mnist()

    if not exp_family != EinsumNetwork.NormalArray:
        train_x /= 255.
        test_x /= 255.
        train_x -= .5
        test_x -= .5

    # validation split
    valid_x = train_x[-10000:, :]
    train_x = train_x[:-10000, :]
    valid_labels = train_labels[-10000:]
    train_labels = train_labels[:-10000]

    # pick the selected classes
    if classes is not None:
        train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
        valid_x = valid_x[np.any(np.stack([valid_labels == c for c in classes], 1), 1), :]
        test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

    train_x = torch.from_numpy(train_x).to(torch.device(device))
    valid_x = torch.from_numpy(valid_x).to(torch.device(device))
    test_x = torch.from_numpy(test_x).to(torch.device(device))

    # Make Einsum Network
    ####################################################################################
    structure = 'poon-domingos'
    height = 28
    width = 28
    
    if structure == 'poon-domingos':
        pd_delta = [[height / d, width / d] for d in pd_pieces]
        graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
    elif structure == 'binary-trees':
        graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)
    else:
        raise AssertionError("Unknown Structure")

    args = EinsumNetwork.Args(
            num_var=train_x.shape[1],
            num_dims=1,
            num_classes=1,
            num_sums=K,
            num_input_distributions=K,
            exponential_family=exp_family,
            exponential_family_args=exponential_family_args,
            online_em_frequency=online_em_frequency,
            online_em_stepsize=online_em_stepsize)

    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    print(einet)

    # Train
    ######################################
    train_N = train_x.shape[0]
    valid_N = valid_x.shape[0]
    test_N = test_x.shape[0]

    for epoch_count in range(num_epochs):
        
        start = time.time()
        ##### evaluate
        einet.eval()
        train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
        valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
        test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
        print("[{}]   train LL {}   valid LL {}   test LL {}".format(
            epoch_count,
            train_ll / train_N,
            valid_ll / valid_N,
            test_ll / test_N))
        einet.train()
        #####

        idx_batches = torch.randperm(train_N, device=device).split(batch_size)

        total_ll = 0.0
        for idx in idx_batches:
            batch_x = train_x[idx, :]
            outputs = einet.forward(batch_x)
            ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()

            einet.em_process_batch()
            total_ll += log_likelihood.detach().item()

        einet.em_update() 
        print(f'Time elapsed in epoch {epoch_count} is {np.round(time.time()-start, 2)} seconds.')

    if fashion_mnist:
        model_dir = save_dir
    else:
        model_dir = save_dir
    utils.mkdir_p(model_dir)


    # save model
    #graph_file = os.path.join(model_dir, "einet.pc")
    #Graph.write_gpickle(graph, graph_file)
    #print("Saved PC graph to {}".format(graph_file))
    model_file = os.path.join(model_dir, "einet.mdl")
    torch.save(einet, model_file)
    print("Saved model to {}".format(model_file))

# train_model('../models/einet/demo_mnist_test/', None, EinsumNetwork.NormalArray, K = 25, pd_pieces = [4,7], fashion_mnist = False, num_epochs = 20, batch_size = 100, online_em_frequency = 1, online_em_stepsize = 0.05)
# train_model('../models/einet/demo_mnist_test/', [7], EinsumNetwork.NormalArray, K = 5, pd_pieces = [4], fashion_mnist = False, num_epochs = 1, batch_size = 10, online_em_frequency = 1, online_em_stepsize = 0.05)

def image_expectation(model, num_pixels, batch_size, K = 15, gaussian = True, means = None, save = False, save_dir = None):
    '''
    This function works for non-Gaussian distributions, but not sure that the lower-level
    functions are compatible with them yet.

    Best if batch_size divides num_pixels - though shouldn't be hard to fix.
    '''

    dist_layer = model.einet_layers[0]
    
    # Get distribution means
    if means is not None:
        means = means
    elif gaussian:
        means = dist_layer.ef_array.params.squeeze()[:,:,0].unsqueeze(2)
    else:
        means = dist_layer.ef_array.params.squeeze().unsqueeze(2)

    # Initialise expectation tensor
    expectations = torch.zeros(784)

    for batch_no in range(num_pixels // batch_size):
        print(f'Batch {batch_no}')
        x = torch.zeros((batch_size, num_pixels, K, 1))

        for idx in range(batch_size):
            pixel = batch_size * batch_no + idx
            x[idx,pixel,:] = means[pixel,:]
        
        expectations[batch_no*batch_size:(batch_no + 1)*batch_size] = model.expectation(x).detach().squeeze()

    if save and save_dir is not None:
        utils.mkdir_p(save_dir)
        utils.save_image_stack(torch.reshape(expectations,(1,28,28)),\
         1, 1, filename = save_dir, margin_gray_val=0.)
        print(f'Images saved to {save_dir}')
    elif save:
        print('Need name of directory to save images to...')

    return expectations

#print(image_expectation('../models/einet/demo_mnist_5/einet.mdl', 784, 28, 15, True))
#print(image_expectation('../models/einet/demo_mnist_3/einet.mdl', 784, 28, 15, False))

def get_variance(phi, dist_layer):
    '''Only works for NormalArray'''
    theta = dist_layer.ef_array.expectation_to_natural(phi)
    variance = theta[:,:,:,1] * -2
    variance = torch.reciprocal(variance)

    return variance

#print(get_variance(phi))


def gaussian_product(phi, y, epsilon, dist_layer):
    '''
    Returns the mean and variance of the product of two Gaussians
    means and variance 1's refer to statistics of distribution
    means and variance 2's refer to statistics of denoising Gaussian
    '''

    # Get statistics of the distribtuion units
    variance1 = get_variance(phi, dist_layer)
    mean1 = phi[:,:,:,0]
    
    # Calculate combined mean
    mean_num = epsilon * mean1 + torch.mul(y, variance1)
    denom = torch.add(variance1, epsilon)
    mean = mean_num / denom

    # Calculate combined variances
    var_num = variance1 * epsilon
    var = var_num / denom
    
    return mean, var

# gaussian_product(phi, torch.ones(784,15,1), 2)

def denoising_expectation(model_file, noisy_image, epsilon, num_pixels, batch_size, K = 15, gaussian = True, save = False, save_dir = None):
    ''' Makes use of newly improved expectation function'''

    # Get model
    einet = torch.load(model_file)
    dist_layer = einet.einet_layers[0]

    # Format noisy image
    y = torch.reshape(noisy_image,(1,784)).squeeze()
    y = torch.transpose(y.repeat(K,1),0,1).unsqueeze(2)

    # Distibution parameters
    phi = dist_layer.ef_array.params
    mean, var = gaussian_product(phi, y, epsilon, dist_layer)

    return image_expectation(einet, num_pixels, batch_size, K = K, gaussian = gaussian, means = mean, save = save, save_dir = save_dir)
    
#utils.mkdir_p('../expectation/demo_mnist/1/')

#for i in range(10):
#    noisy_image = load_images('../blur2/', True)[i,:,:,:]
#    denoising_expectation('../models/einet/demo_mnist/einet.mdl',noisy_image, 0.01, 784, 28, K = 10, save = True, save_dir = f'../expectation/demo_mnist/1/denoised_{i}.png')