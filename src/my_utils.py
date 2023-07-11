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

#test = load_images('../samples/demo_mnist_3/')
#print(type(test), test.size(), test[999])

def FID(dir_real, dir_gen, features = 2048):

    fid = FrechetInceptionDistance(feature=features)

    # Load images as tensors
    imgs_true = load_images(dir_real).type(torch.uint8)[:127]
    imgs_false = load_images(dir_gen).type(torch.uint8)[:127]

    print('Images have been loaded in.')

    # Calculate and print the FID distance
    fid.update(imgs_true, real=True)
    print('FID object updated with true images.')
    fid.update(imgs_false, real=False)
    return fid.compute()

#print(FID('../split_datasets/mnist3/part1', '../samples/demo_mnist_3/'))
#print(FID('../split_datasets/mnist7/part_1', '../split_datasets/mnist3/part_2'))


def gaussian_noise(img_dir, gs = True, sigma = 0.05, save = False, save_dir = None):
    
    # Load the images and store them as a tensor
    x = load_images(img_dir, grey_scale = gs)

    shape = x.size() # Save the shape of image tensors for later
    x = torch.flatten(x, start_dim = 1)
    d = x.size()[1]

    # Noise
    means = np.zeros(d)
    cov = sigma * np.eye(d)

    w = np.random.multivariate_normal(means, cov, shape[0])
    y = x + w 

    if save and save_dir is not None:
        utils.mkdir_p(save_dir)
        for i in range(len(y)):
            utils.save_image_stack(torch.reshape(y[i],(1,shape[2],shape[2])), 1, 1, os.path.join(save_dir, f'noise_image_{i}.png'), margin_gray_val=0.)
        print(f'Images saved to {save_dir}')
    elif save:
        print('Need name of directory to save images to...')

    return y

#gaussian_noise('../blur/', save = True, save_dir = '../blur2/2')


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

#sampling(10, '../models/einet/demo_mnist_3/', '../samples/demo_mnist_3/1/', True)

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

#split_mnist(None, save = True, save_dir = '../split_datasets/mnist')



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
