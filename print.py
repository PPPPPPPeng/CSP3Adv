from backbones.insightface import *  # insightface-get_embd()
from utils.util import *
from utils.attack import *
from scipy import spatial

from skimage import color
import cv2
from backbones.unet import unet
from backbones.MobileFaceNet import mobilefacenet
from utils.mmd import *
import random
from backbones.inception_resnet_v1 import inception_resnet_v1  # FaceNet
import sphere_torch  # SphereFace
import os
import cv2
from skimage import color
from advfaces import AdvFaces
from utils.imageprocessing import preprocess

from torchvision import utils as vutils
import datetime
from utils.differentiableJPEG import *
from skimage.restoration import denoise_tv_bregman as _denoise_tv_bregman
#
# temp = Image.open(r'/home/yl/PycharmProjects/sumail/APF/论文选图/original.png').convert('RGB')
# temp.save(r'/home/yl/PycharmProjects/sumail/APF/论文选图/original-75.jpeg',format='JPEG',quality=75)
# temp=misc.imread(r'/home/yl/PycharmProjects/sumail/APF/论文选图/original.png')
# j = misc.imread(r'/home/yl/PycharmProjects/sumail/APF/论文选图/original-75.jpeg')
# kernel = gkern(3, 7.5).astype(np.float64)  # sigma=(k-1)/6 ？
# stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
# stack_kernel = np.expand_dims(stack_kernel, 3)
# temp_gray = color.rgb2gray(temp)
# g =tf.nn.depthwise_conv2d(temp, stack_kernel, strides=[1, 1, 1], padding='SAME') # Guassian filter
# j_gray = color.rgb2gray(j)
# # temp=temp/127.5 -1.0
# # j=j/127.5 - 1.0
# print(temp_gray)
# print("__________________________")
# print(j_gray)
# print("__________________________")
# print(cv2.absdiff(temp_gray,j_gray))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  # print(kernel)
  return kernel
s=gkern(3,2.9)
print(s)

#
# with tf.Session(config=config) as sess:
#   test1 = tf.constant([[1.0,1.0,1.0],[1.0,1.0,1.0]])
#   test2 = tf.constant([[1.0,1.0,5.0],[1.0,1.0,1.0]])
#   # print(test1)
#   # test2 =
#   err = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(test1.eval(),test2.eval())))) #   L2 mean
#   print("error:",err.eval())
#   err = tf.reduce_mean(tf.abs(tf.subtract(test1.eval(), test2.eval())))  # L1 mean *
#   print("error:", err.eval())