"""Functions for image processing
"""


import sys
import os
import math
import random
import numpy as np
from scipy import misc
import cv2


# Calulate the shape for creating new array given (h,w)
def get_new_shape(images, size=None, n=None):
    shape = list(images.shape)
    if size is not None:
        h, w = tuple(size)
        shape[1] = h
        shape[2] = w
    if n is not None:
        shape[0] = n
    shape = tuple(shape)
    return shape

def random_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)
    assert (_h>=h and _w>=w)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    y = np.random.randint(low=0, high=_h-h+1, size=(n))
    x = np.random.randint(low=0, high=_w-w+1, size=(n))

    for i in range(n):
        images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]

    return images_new

def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def random_flip(images):
    images_new = images.copy()
    flips = np.random.rand(images_new.shape[0])>=0.5
    
    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])

    return images_new

def flip(images):
    images_new = images.copy()
    for i in range(images_new.shape[0]):
        images_new[i] = np.fliplr(images[i])

    return images_new

def resize(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        images_new[i] = misc.imresize(images[i], (h,w))

    return images_new

def padding(images, padding):
    n, _h, _w = images.shape[:3]
    if len(padding) == 2:
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
    else:
        pad_t, pad_b, pad_l, pad_r = tuple(padding)
       
    size_new = (_h + pad_t + pad_b, _w + pad_l + pad_r)
    shape_new = get_new_shape(images, size_new)
    images_new = np.zeros(shape_new, dtype=images.dtype)
    images_new[:, pad_t:pad_t+_h, pad_l:pad_l+_w] = images

    return images_new

def standardize_images(images, standard):
    if standard=='mean_scale':
        mean = 127.5
        std = 128.0
    elif standard=='scale':
        mean = 0.0
        std = 255.0
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new



def random_shift(images, max_ratio):
    n, _h, _w = images.shape[:3]
    pad_x = int(_w * max_ratio) + 1
    pad_y = int(_h * max_ratio) + 1
    images_temp = padding(images, (pad_y, pad_x))
    images_new = images.copy()

    shift_x = (_w * max_ratio * np.random.rand(n)).astype(np.int32)
    shift_y = (_h * max_ratio * np.random.rand(n)).astype(np.int32)

    for i in range(n):
        images_new[i] = images_temp[i, pad_y+shift_y[i]:pad_y+shift_y[i]+_h, pad_x+shift_x[i]:pad_x+shift_x[i]+_w]

    return images_new    
    
def random_rotate(images, max_degree):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()

    degree = max_degree * np.random.rand(n)
    tokens = np.random.rand(n) >= 0.5
    for i in range(n):
        M = cv2.getRotationMatrix2D((_w/2, _h/2), int(degree[i]), 1)

        for img in images:
            if tokens[i]:
                M = cv2.getRotationMatrix2D((_w / 2, _h / 2), int(degree[i]), 1)
                img = cv2.warpAffine(img, M, (_w, _h))
        # images = [cv2.warpAffine(img, M, (_w, _h)) for img in images]

    return images_new


def random_blur(images, blur_type, max_size):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    
    kernel_size = max_size * np.random.rand(n)
    
    for i in range(n):
        size = int(kernel_size[i])
        if size > 0:
            if blur_type == 'motion':
                kernel = np.zeros((size, size))
                kernel[int((size-1)/2), :] = np.ones(size)
                kernel = kernel / size
                img = cv2.filter2D(images[i], -1, kernel)
            elif blur_type == 'gaussian':
                size = size // 2 * 2 + 1
                img = cv2.GaussianBlur(images[i], (size,size), 0)
            else:
                raise ValueError('Unkown blur type: {}'.format(blur_type))
            images_new[i] = img

    return images_new
    
def random_noise(images, stddev, min_=-1.0, max_=1.0):

    noises = np.random.normal(0.0, stddev, images.shape)
    images_new = np.maximum(min_, np.minimum(max_, images + noises))
        
    return images_new

def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    ratios = min_ratio + (1-min_ratio) * np.random.rand(n)

    for i in range(n):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i,:h,:w] = misc.imresize(images[i], (h,w))
        images_new[i] = misc.imresize(images_new[i,:h,:w], (_h,_w))
        
    return images_new

def random_interpolate(images):
    _n, _h, _w = images.shape[:3]
    nd = images.ndim - 1
    assert _n % 2 == 0
    n = int(_n / 2)

    ratios = np.random.rand(n,*([1]*nd))
    images_left, images_right = (images[np.arange(n)*2], images[np.arange(n)*2+1])
    images_new = ratios * images_left + (1-ratios) * images_right
    images_new = images_new.astype(np.uint8)

    return images_new
    
def expand_flip(images):
    '''Flip each image in the array and insert it after the original image.'''
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, n=2*_n)
    images_new = np.stack([images, flip(images)], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new

def five_crop(images, size):  #??????????????????????????????????????????????????? 5 ?????????
    _n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert h <= _h and w <= _w

    shape_new = get_new_shape(images, size, n=5*_n)
    images_new = []
    images_new.append(images[:,:h,:w])
    images_new.append(images[:,:h,-w:])
    images_new.append(images[:,-h:,:w])
    images_new.append(images[:,-h:,-w:])
    images_new.append(center_crop(images, size))
    images_new = np.stack(images_new, axis=1).reshape(shape_new)
    return images_new

def ten_crop(images, size):  # ??? FiveCrop ???????????????????????? 5 ?????????????????????????????????????????????10???
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, size, n=10*_n)
    images_ = five_crop(images, size)
    images_flip_ = five_crop(flip(images), size)
    images_new = np.stack([images_, images_flip_], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new
    
    

register = { # ????????????????????????
    'resize': resize, # 160,160
    'padding': padding,
    'flip': flip,
    'random_crop': random_crop,
    'center_crop': center_crop,
    'random_flip': random_flip, # trainset
    'standardize': standardize_images,  # trainset-mean/testset
    'random_shift': random_shift,
    'random_interpolate': random_interpolate,
    'random_rotate': random_rotate,
    'random_blur': random_blur,
    'random_noise': random_noise,
    'random_downsample': random_downsample,
    'expand_flip': expand_flip,
    'five_crop': five_crop,
    'ten_crop': ten_crop,
}

def preprocess(images, config, is_training=False):
    ''' Legacy function. Equaivalent to batch_process but it uses config module as input '''
    # Process images
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert proc_name in register, \
            "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:,:,:,None]
    return images
        

def batch_process(images, proc_funcs, channels=3):
    # Load images first if they are file paths
    if type(images[0]) == str:
        image_paths = images
        images = []
        assert (channels==1 or channels==3)
        mode = 'RGB' if channels==3 else 'I'
        for image_path in image_paths:
            images.append(misc.imread(image_path, mode=mode))
        images = np.stack(images, axis=0)
    else:
        assert type(images[0]) is np.array, \
            'Illegal input type for images: {}'.format(type(images[0]))

    # Process images
    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert proc_name in register, \
            "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:,:,:,None]
    return images
        
