import math
import numpy as np
import scipy.io as sio
from skimage.transform import resize
import copy
from random import *
from PIL import Image

def ReadMap(mapPath,name):
    """
    Load the density map from matfile.
    """
    map_mat_data = sio.loadmat(mapPath)
    map_data = map_mat_data[name]
    map_data = map_data.astype('float32')
    return map_data

def get_depth_input(im_np, factor):

    h0, w0 = im_np.shape[:2]
    h, w = int(math.ceil(h0 / float(factor))), int(math.ceil(w0 / float(factor)))
    

    # Upsampling the low resolution depth image by bicubic interpolation
    #order_dict = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}
    lowres = resize(im_np, (h, w), order=0, mode='reflect', clip=False, preserve_range=True, anti_aliasing=True)
    highres = resize(lowres, (h0, w0), order=0, mode='reflect', clip=False, preserve_range=True)

    return highres

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    h,w  = gray.shape
    gray = gray.reshape(h,w,1)
    return gray

def load_data_pairs(img_path, depth_path, factor=4):
    """load all volume pairs"""
    guide_img = ReadMap(img_path,'images')
    gt_depth = ReadMap(depth_path,'depths')

    guide_img = guide_img.astype('float32')
    gt_depth = gt_depth.astype('float32')

    guide_img = guide_img/255.0
    gt_depth = gt_depth/10.0

    input_depth = get_depth_input(gt_depth, factor)

    guide_img = guide_img.transpose(3, 0, 1, 2)
    input_depth = input_depth.transpose(2, 0, 1)
    gt_depth = gt_depth.transpose(2, 0, 1)

    return guide_img, input_depth, gt_depth

def get_batch_patches(rand_guide, rand_input, rand_gt, patch_dim, batch_size):

    if np.random.random() > 0.5:
        rand_guide=np.fliplr(rand_guide)
        rand_input=np.fliplr(rand_input)
        rand_gt=np.fliplr(rand_gt)
    
    if patch_dim[2]<3:
        rand_guide = rgb2gray(rand_guide)
    w, h, c = rand_guide.shape

    patch_width = int(patch_dim[0])
    patch_heigh = int(patch_dim[1])

    batch_guide = np.zeros([batch_size, patch_width, patch_heigh, c]).astype('float32')
    batch_input = np.zeros([batch_size, patch_width, patch_heigh, 1]).astype('float32')
    batch_gt = np.zeros([batch_size, patch_width, patch_heigh, 1]).astype('float32')

    for k in range(batch_size):
        # randomly select a box anchor
        w_rand = randint(0, w - patch_width)
        h_rand = randint(0, h - patch_heigh)

        pos = np.array([w_rand, h_rand])
        # crop
        batch_guide[k, :, :, :] = copy.deepcopy(rand_guide[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh, :])
        batch_input[k, :, :, 0] = copy.deepcopy(rand_input[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh])
        batch_gt[k, :, :, 0] = copy.deepcopy(rand_gt[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh])

    return batch_guide, batch_input, batch_gt
