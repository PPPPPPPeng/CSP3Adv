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
from advfaces import AdvFaces
from utils.imageprocessing import preprocess

from torchvision import utils as vutils
import datetime
from utils.differentiableJPEG import *
from skimage.restoration import denoise_tv_bregman as _denoise_tv_bregman

#对于 JPEG0-100 匹配一个吻合的高斯滤波器
# 对于每个压缩质量因子的 JPEG，sigma 从 0-5，step=0.01  窗口大小 3-21，step=2


X = create_lfw_npy_samepair()  # samepair测试集图片
num_test = X.shape[0]
batch_size = 100
batch = 0
x_batch = X[batch*batch_size:(batch+1)*batch_size]
print(x_batch.shape)
x_jpeg = np.zeros((100,112,112,3))
print(x_jpeg.shape)
# x_jpeg = x_batch # 出问题
saveplace = '/home/yl/PycharmProjects/sumail/APF/论文选图/temp'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

for q in range(84,100):#JPEG 0-100
    err = 0
    best_sigma = 0
    best_res = float("inf")
    best_n = 0
    for index in range(batch_size): # 准备 JPEG 图片
        misc.imsave(saveplace + '/{}.png'.format(index), x_batch[index])
        x_batch[index] = misc.imread(saveplace + '/{}.png'.format(index))
        # jpeg
        # temp = Image.open(saveplace+'/{}.png'.format(index)).convert('RGB')
        # temp.save(saveplace+'/{}.jpeg'.format(index),format='JPEG',quality=q)
        # WEBP
        t = cv2.imread(saveplace + '/{}.png'.format(index))
        cv2.imwrite(saveplace + '/{}.webp'.format(index), t, [cv2.IMWRITE_WEBP_QUALITY, q])

        x_jpeg[index] = misc.imread(saveplace + '/{}.webp'.format(index)) # todo
        # print("jpeg after:",x_jpeg[index])
        os.remove(saveplace + '/{}.webp'.format(index)) # WEBP
        # os.remove(saveplace + '/{}.jpeg'.format(index)) # JPEG
        os.remove(saveplace + '/{}.png'.format(index))  #

    for s in np.arange(1,8,0.1): # sigma[0,] step=0.1
        for n in range(3,11,2): # kernelsize[3,9]
            tf.reset_default_graph()
            with tf.Session(config=config) as sess:
                kernel = gkern(n, s).astype(np.float64)  # sigma=(k-1)/6 ？
                # print("kernel:",kernel)
                stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
                stack_kernel = np.expand_dims(stack_kernel, 3)
                g =tf.nn.depthwise_conv2d(x_batch, stack_kernel, strides=[1, 1, 1, 1], padding='SAME') # Guassian filter
                # clip
                # print("g before:",g.eval())
                g_gray = tf.image.rgb_to_grayscale(g)
                jpeg_gray = tf.image.rgb_to_grayscale(x_jpeg)
                for img in range(batch_size):
                    # print(jpeg_gray[img].eval()-g_gray[img].eval())
                    # print("error:",tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(g_gray[img].eval(), jpeg_gray[img].eval())))).eval() )
                    # print("img:",x_batch[img]) #
                    # print("g:", g_gray[img].eval())
                    # print("jpeg_gray", jpeg_gray[img].eval())

                    # err += tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(g_gray[img].eval(), jpeg_gray[img].eval())))) #   L2 mean

                    # err += tf.reduce_mean(tf.square(tf.subtract(g_gray[img].eval(), jpeg_gray[img].eval()))) #   L2² mean
                    # err += tf.reduce_sum(tf.square(g_gray[img].eval() - jpeg_gray[img].eval()))  # L2-² sum
                    # err += tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(g_gray[img].eval(), jpeg_gray[img].eval()))))  # L2 sum

                    err += tf.reduce_mean(tf.abs(tf.subtract(jpeg_gray[img].eval(), g_gray[img].eval()))) #   L1 mean *

                    # err += tf.reduce_sum(tf.abs(tf.subtract(jpeg_gray[img].eval(), g_gray[img].eval()))) # L1 sum
                    if (img==(batch_size-1)):
                        print("Quality: {} sigma : {}  kernerl_size: {} [{}/{}] dist:".format(q, s, n, img, batch_size),err.eval())
                # sess.run(err)
                if err.eval()<best_res:
                    best_res = err.eval()
                    best_n = n
                    best_sigma = s
                err = 0
            tf.get_default_graph().finalize()
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("当前时间 {}  JPEG[{}] best kernel_size = {}, best sigma = {}, result = {}".format(time, q,best_n,best_sigma,best_res))
    with open('/home/yl/PycharmProjects/sumail/APF/JPEG_results','a') as f:
        f.writelines("当前时间 {}  JPEG[{}] best kernel_size = {}, best sigma = {}, result = {}".format(time, q,best_n,best_sigma,best_res)+'\n')

