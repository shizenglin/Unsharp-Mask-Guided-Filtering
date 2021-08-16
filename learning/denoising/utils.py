import os
import math
import numpy as np
import scipy.io as sio
from skimage.transform import rescale
import copy
from random import *
from PIL import Image
from glob import glob

def SaveImage(predicted_img, save_path):

    img = Image.fromarray(np.array(predicted_img*255.0).astype('uint8'))

    #rgbimg = Image.new("RGBA", img.shape)
    #rgbimg.paste(img)
    img.save(save_path+'.jpg')

def ReadImage(imPath,mirror = False,scale=1.0):
    """
    Read gray images.
    """
    imArr = np.array(Image.open(imPath))#.convert('L'))
    #print imArr
    if(scale!=1):
        imArr = rescale(imArr, scale,preserve_range=True)#
        #print imArr
        #time.sleep(10)
    if (len(imArr.shape)<3):
        imArr = imArr[:,:,np.newaxis]
        imArr = np.tile(imArr,(1,1,3))

    return imArr

def ReadMap(mapPath,name):
    """
    Load the density map from matfile.
    """
    map_mat_data = sio.loadmat(mapPath)
    map_data = map_mat_data[name]
    map_data = map_data.astype('float32')
    return map_data

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    h,w  = gray.shape
    gray = gray.reshape(h,w,1)
    return gray

def load_data_pairs(img_path, depth_path):
    """load all volume pairs"""
    guide_img = ReadMap(img_path,'images')
    guide_img = guide_img.transpose(3, 0, 1, 2)
    guide_img = guide_img[:1000]
    gt_depth = ReadMap(depth_path,'depths')
    gt_depth = gt_depth.transpose(2, 0, 1)
    gt_depth = gt_depth[:1000]

    guide_img = guide_img.astype('float32')
    gt_depth = gt_depth.astype('float32')

    guide_img = guide_img/255.0
    gt_depth = gt_depth/10.0

    return guide_img, gt_depth

def load_data_pairs_test(img_path, depth_path, test_name='nyu'):
    if test_name == 'nyu':
        guide_img = ReadMap(img_path,'images')
        guide_img = guide_img.transpose(3, 0, 1, 2)
        guide_img = guide_img[1000:]
        gt_depth = ReadMap(depth_path,'depths')
        gt_depth = gt_depth.transpose(2, 0, 1)
        gt_depth = gt_depth[1000:]

        guide_img = guide_img.astype('float32')
        gt_depth = gt_depth.astype('float32')

        guide_img = guide_img/255.0
        gt_depth = gt_depth/10.0

        n,h,w = gt_depth.shape
        input_depth_15 = gt_depth+np.random.randn(n,h,w)*(15.0/255.0)
        input_depth_25 = gt_depth+np.random.randn(n,h,w)*(25.0/255.0)
        input_depth_50 = gt_depth+np.random.randn(n,h,w)*(50.0/255.0)

    return guide_img, input_depth_15, input_depth_25, input_depth_50, gt_depth
def get_batch_patches(rand_guide, rand_gt, patch_dim, batch_size):
    #print rand_img.shape
    #rand_guide = rgb2gray(rand_guide)
    if np.random.random() > 0.5:
        rand_guide=np.fliplr(rand_guide)
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

        gt_temp = copy.deepcopy(rand_gt[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh])
        batch_input[k, :, :, 0] = gt_temp + np.random.randn(patch_width,patch_heigh)*np.random.uniform(0,0.21)
        batch_gt[k, :, :, 0] = gt_temp

    return batch_guide, batch_input, batch_gt
