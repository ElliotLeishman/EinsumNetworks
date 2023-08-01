'''
Not really worked - need to think more about how to blur better?
'''

import numpy as np
import torch
import utils
import my_utils
import os

images = my_utils.load_images('../blur/', True)

test_image = torch.reshape(images[0,:],(1,784))

A = torch.diagflat(torch.ones(783), 1) + torch.diagflat(torch.ones(783), -1) 

blurred = A @ torch.transpose(test_image,0,1)

blurred = torch.reshape(blurred, (1,28,28))
utils.save_image_stack(blurred,1, 1, os.path.join('../blur2/', f'image_1.png'), margin_gray_val=0.)


