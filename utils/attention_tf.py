from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torchvision import utils as vutils
import tensorflow as tf
# import tensorflow.contrib.slim as slim
import tf_slim as slim
def se_block(residual, name, ratio=8): # SE ATTENTION
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = residual.get_shape()[-1]
        squeeze = tf.reduce_mean(residual, axis=[1, 2], keepdims=True) # 全局平均池化Global average pooling
        # squeeze = tf.nn.max_pool(residual,ksize=[1,2,2,1],strides = [1,1,1,1],padding='VALID')  # MaxPool
        assert squeeze.get_shape()[1:] == (1, 1, channel)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel // ratio,
                                     activation=tf.nn.relu,  # ReLU
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        assert excitation.get_shape()[1:] == (1, 1, channel // ratio)
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,# Sigmoid
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        assert excitation.get_shape()[1:] == (1, 1, channel)
        # top = tf.multiply(bottom, se, name='scale')
        scale = residual * excitation
    return scale


def cbam_block(input_feature, name, ratio=4): # CBAM ATTENTION
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
        print("CBAM Hello")
    return attention_feature


def Coord_block(x,channel, reduction = 32):
    print("Coord Hello")
    def coord_act(x):
        tmpx = tf.nn.relu6(x+3) / 6
        x = x * tmpx
        return x

    x_shape = x.get_shape().as_list()
    [b, h, w, c] = x_shape
    c = channel
    x_h = slim.avg_pool2d(x, kernel_size = [1, w], stride = 1)
    x_w = slim.avg_pool2d(x, kernel_size = [h, 1], stride = 1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    y = tf.concat([x_h, x_w], axis=1)
    mip = max(8, c // reduction) # 8
    y = slim.conv2d(y, mip, (1, 1), stride=1, padding='VALID', normalizer_fn = slim.batch_norm, activation_fn=coord_act,scope='ca_conv1_{}'.format(c),reuse=tf.AUTO_REUSE)

    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    a_h = slim.conv2d(x_h, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv2_{}'.format(c),reuse=tf.AUTO_REUSE)
    a_w = slim.conv2d(x_w, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv3_{}'.format(c),reuse=tf.AUTO_REUSE)

    out = x * a_h * a_w


    return out


def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat