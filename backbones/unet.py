import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from scipy import misc
from utils.attention_tf import (se_block,cbam_block,Coord_block)
attention = [se_block, cbam_block, Coord_block]
universal = np.load('/home/yl/PycharmProjects/sumail/APF/precomputing_perturbations/perturbation_wuqiongfanshu.npy')

channel_list = [64, 128, 256, 512]
channel_num = len(channel_list)
cnt = 0
# ""ResUNet++ architecture in Keras TensorFlow

import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def unet(x_input, scope=None, reuse=None, universal=universal):
    with tf.variable_scope('APF', reuse=reuse):
        x_down = [x_input] # 下采样
        for i, item in enumerate(channel_list): # 下采样channel_list[0:64, 1:128, 2:256, 3:512]
            x = tf.layers.conv2d(
                x_down[i],
                filters=item,
                strides=2,
                kernel_size=[3,3],
                padding='SAME',
                activation=tf.nn.leaky_relu
            )
            x_down.append(x)
        channel_list_reverse = [256, 128, 64]
        x_up = [x_down[-1]] #上采样
        for i, item in enumerate(channel_list_reverse): # 上采样[0:256, 1:128, 2:64]
            x = tf.layers.conv2d_transpose(
                x_up[i],
                filters=item,
                strides=2,
                kernel_size=[3,3],
                padding='SAME',
                activation=tf.nn.leaky_relu
            )


            # x_up.append(tf.concat([x, x_down[-(i+2)] ],axis=-1)) # skip connection
            # x_up.append(tf.concat([x, attention[0](x_down[-(i + 2)], name="skip_{}".format(i))],axis=-1))  # skip connection with SE
            # x_up.append(tf.concat([x, attention[1](x_down[-(i+2)],name = "skip_{}".format(i)) ],axis=-1)) # skip connection with CBAM
            x_up.append(tf.concat([x, attention[2](x_down[-(i + 2)],channel = channel_list[-(i+2)])],axis=-1))  # skip connection with Coord
            # 256-256, 128-128, 64-64


        # g~
        unet_output = tf.layers.conv2d_transpose(x_up[-1],filters=3,strides=2,kernel_size=[3,3],padding='SAME',activation=None)




        # E Module
        # universal = tf.convert_to_tensor(universal, tf.float32)  # universal: (u)
        # scale_alpha = tf.Variable(tf.constant(1.0))  #trainable  DI2FGSM:0.5291966
        # scale_beta = tf.Variable(tf.constant(0.0))    #trainable DI2FGSM:0.010275785
        # print(scale_alpha,scale_beta)
        # universal = universal * scale_alpha + scale_beta
        # noise = tf.layers.conv2d(inputs=universal,filters=3,kernel_size=[1, 1],strides=1,padding='valid',activation=None,use_bias=True)# noise:enhanced universal perturbation(u-hat)
        # output = noise + unet_output # output: enhanced adversarial noise (s)
        # return output  # 使用UAP
        return unet_output # 不使用UAP
