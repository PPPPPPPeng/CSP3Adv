import tensorflow as tf
import numpy as np
import yaml
from utils.mmd import *
from PIL import Image
from random import randint
IMAGE_SHAPE = 112
prob = 0.5
epsilon = 8 / 255. * 2 # 源代码使用的
# epsilon = 8/255.
step_size = 2 / 255. * 2
# step_size = 2 / 255
# niter = 10
niter = 30
bounds = (-1, 1)

config = yaml.load(open('./configs/config_ms1m_100.yaml'),Loader=yaml.FullLoader)
def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  # print(kernel)
  return kernel
# TI
kernel = gkern(3, 3).astype(np.float32) # kernel size
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)
# net = tf.nn.depthwise_conv2d(net, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')  # TI

def FGM(x, dist, eps= epsilon):
    print(' FGM Attack')
    grad = tf.gradients(dist, x)[0]
    # reduc_ind = list(range(1, len(x.get_shape())))
    x_adv = x + eps * grad / (tf.math.sqrt(tf.square(grad)) + 1e-8)
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)
    # if ord==np.inf:
    #         signed_grad = tf.sign(gradient)
    #     elif ord== 1:
    #         reduc_ind = list(range(1, len(model.x_input.get_shape())))
    #         signed_grad = gradient / tf.reduce_sum(tf.abs(gradient),
    #                                            reduction_indices=reduc_ind,
    #                                            keep_dims=True)
    #     elif ord == 2:
    #         reduc_ind = list(range(1,len(model.x_input.get_shape())))
    #         signed_grad = gradient / tf.sqrt(tf.reduce_sum(tf.square(gradient),
    #                                                        reduction_indices=reduc_ind,
    #                                                        keep_dims=True))

def FGSM(x, dist, eps=epsilon):
    print(' FGSM Attack')
    x_adv = x + eps * tf.sign(tf.gradients(dist, x)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)

def FGSM2(x, model_function, dist_function, perturbation_multiplier=1): # 5次的IFGSM
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < 5

    def loop_body(index, adv_images):
        adv_images_di = input_diversity(adv_images)
        tmp_embd = model_function(adv_images_di)
        dist = dist_function(tmp_embd)
        grad = tf.gradients(dist, adv_images)[0]
        perturbation = step_size * tf.sign(grad)
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    _, result = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars,
        back_prop=False,
        parallel_iterations=1)
    return result #


def IFGSM(x, model_function, dist_function, perturbation_multiplier=1):# 标准的I-FGSM
    print(' IFGSM Attack')
    print('total iteration:',niter)
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < niter

    def loop_body(index, adv_images):
        tmp_embd = model_function(adv_images)
        dist = dist_function(tmp_embd)
        grad = tf.gradients(dist, adv_images)[0]
        #  stepsize small or large
        # s = epsilon / niter
        # perturbation = s * tf.sign(grad)
        perturbation = step_size * tf.sign(grad)
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    _, result = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars,
        back_prop=False,
        parallel_iterations=1)
    return result

def TIFGSM(x, model_function, dist_function, perturbation_multiplier=1):# 标准的TI-FGSM todo
    print(' TIFGSM Attack')
    print('total iteration:',niter)
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < niter

    def loop_body(index, adv_images):
        tmp_embd = model_function(adv_images)
        dist = dist_function(tmp_embd)
        grad = tf.gradients(dist, adv_images)[0]
        perturbation = step_size * tf.sign(grad)
        perturbation =  tf.nn.depthwise_conv2d(perturbation, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')  # TI
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    _, result = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars,
        back_prop=False,
        parallel_iterations=1)
    return result

def PGD(x, model_function, dist_function, perturbation_multiplier=1):#  todo
    print(' PGD Attack')
    print('total iteration:',niter)
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x + tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < niter

    def loop_body(index, adv_images):
        tmp_embd = model_function(adv_images)
        dist = dist_function(tmp_embd)
        grad = tf.gradients(dist, adv_images)[0]
        s = epsilon / niter
        perturbation = s * tf.sign(grad) # small stepsize
        perturbation =  tf.nn.depthwise_conv2d(perturbation, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')  # TI
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    _, result = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars,
        back_prop=False,
        parallel_iterations=1)
    return result

def DI2FGSM(x, model_function, dist_function, perturbation_multiplier=1): # DI2-FGSM相比IFGSM +数据增强操作，增加了三种数据变换
    print(' DI2FGSM Attack')
    print('total iteration:', niter)
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < niter

    def loop_body(index, adv_images):
        adv_images_di = input_diversity(adv_images) # new
        tmp_embd = model_function(tf.reshape(adv_images_di, [-1, 112, 112, 3]))


        dist = dist_function(tmp_embd)
        grad = tf.gradients(dist, adv_images)[0]

        perturbation = step_size * tf.sign(grad)
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    _, result = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars,
        back_prop=False,
        parallel_iterations=1)
    return result
def TDI2FGSM(x, model_function, dist_function, perturbation_multiplier=1): #
    print(' DI2FGSM Attack')
    print('total iteration:', niter)
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < niter

    def loop_body(index, adv_images):
        adv_images_di = input_diversity(adv_images) # new
        tmp_embd = model_function(tf.reshape(adv_images_di, [-1, 112, 112, 3]))


        dist = dist_function(tmp_embd)
        grad = tf.gradients(dist, adv_images)[0]
        grad = tf.nn.depthwise_conv2d(grad, stack_kernel, strides=[1, 1, 1, 1], padding='SAME') # guassian filter

        perturbation = step_size * tf.sign(grad)
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    _, result = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars,
        back_prop=False,
        parallel_iterations=1)
    return result

def MIFGSM(x, model_function, dist_function, momentum = 0.9, perturbation_multiplier=1): # 标准的MI-FGSM
    print(' MIFGSM Attack')
    print('total iteration:', niter)
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    start_grad = tf.zeros(tf.shape(x))
    loop_vars = [0, start_x, start_grad]


    def loop_cond(index, _, __):
        return index < niter

    def loop_body(index, adv_images, grad):
        tmp_embd = model_function(adv_images)
        dist = dist_function(tmp_embd)
        noise = tf.gradients(dist, adv_images)[0] #current gradient
        noise = noise / tf.reduce_mean(tf.abs(noise), axis=0, keep_dims=True)
        noise = momentum * grad + noise
        perturbation = step_size * tf.sign(noise)

        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images, noise # noise = accumulated momentum

    with tf.control_dependencies([start_x]):
        _, result, _ = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars,
            back_prop=False,
            parallel_iterations=1)
    return result

def NIFGSM(x, model_function, dist_function, momentum = 0.9, perturbation_multiplier=1):
    print(' NIFGSM Attack')
    print('total iteration:', niter)
    alpha = epsilon/niter

    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x  # + tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    start_grad = tf.zeros(tf.shape(x))
    loop_vars = [0, start_x, start_grad]

    def loop_cond(index, _, __):
        return index < niter

    def loop_body(index, adv_images, grad):
        x_nes = adv_images + momentum * alpha * grad  # todo
        tmp_embd = model_function(x_nes)
        dist = dist_function(tmp_embd)
        noise = tf.gradients(dist, adv_images)[0]
        noise = noise / tf.reduce_mean(tf.abs(noise), axis=0, keep_dims=True)
        noise = momentum * grad + noise
        perturbation = step_size * tf.sign(noise)

        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images, noise  # noise?

    with tf.control_dependencies([start_x]):
        _, result, _ = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars,
            back_prop=False,
            parallel_iterations=1)
    return result

def SINIFGSM(x, model_function, dist_function, momentum = 1, perturbation_multiplier=1):
    print(' SI-NIFGSM Attack')
    print('total iteration:', niter)
    alpha = epsilon/niter

    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x  # + tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    start_grad = tf.zeros(tf.shape(x))
    loop_vars = [0, start_x, start_grad]

    def loop_cond(index, _, __):
        return index < niter

    def loop_body(index, adv_images, grad):
        x_nes = adv_images + momentum * alpha * grad
        x_nes_2 = 1/2 * x_nes
        x_nes_4 = 1 / 4 * x_nes
        x_nes_8 = 1 / 8 * x_nes
        x_nes_16 = 1 / 16 * x_nes
        tmp_embd = model_function(x_nes)
        tmp_embd_2 = model_function(x_nes_2)
        tmp_embd_4 = model_function(x_nes_4)
        tmp_embd_8 = model_function(x_nes_8)
        tmp_embd_16 = model_function(x_nes_16)
        dist = dist_function(tmp_embd)
        dist += dist_function(tmp_embd_2)
        dist += dist_function(tmp_embd_4)
        dist += dist_function(tmp_embd_8)
        dist += dist_function(tmp_embd_16)
        noise = tf.gradients(dist, adv_images)[0]
        noise = noise / tf.reduce_mean(tf.abs(noise), axis=0, keep_dims=True)
        noise = momentum * grad + noise
        perturbation = step_size * tf.sign(noise)

        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images, noise  # noise?

    with tf.control_dependencies([start_x]):
        _, result, _ = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars,
            back_prop=False,
            parallel_iterations=1)
    return result

def MDI2FGSM(x, model_function, dist_function, momentum = 0.9, perturbation_multiplier=1):# M-DI2-FGSM相比MIFGSM +数据增强操作，增加了三种数据变换
    print(' MDI2FGSM Attack')
    print('total iteration:', niter)
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    start_grad = tf.zeros(tf.shape(x))
    loop_vars = [0, start_x, start_grad]


    def loop_cond(index, _, __):
        return index < niter

    def loop_body(index, adv_images, grad):
        adv_images_di = input_diversity(adv_images)
        tmp_embd= model_function(adv_images_di)
        dist = dist_function(tmp_embd)
        noise = tf.gradients(dist, adv_images)[0]

        noise = noise / tf.reduce_mean(tf.abs(noise), axis=0, keep_dims=True)
        noise = momentum * grad + noise
        perturbation = step_size * tf.sign(noise)

        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images, noise

    with tf.control_dependencies([start_x]):
        _, result, _ = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars,
            back_prop=False,
            parallel_iterations=1)
    return result

## image_resize值应该是下一个网络输入的tenor的shape[0]
def input_scaled(input_tensor):
    rnd = tf.random_uniform((), int(IMAGE_SHAPE-2), IMAGE_SHAPE, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = IMAGE_SHAPE - rnd
    w_rem = IMAGE_SHAPE - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], IMAGE_SHAPE, IMAGE_SHAPE, 3))

    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda: padded, lambda: input_tensor)
#

def input_rotate(input_tensor):
    random_angles = tf.random.uniform(shape=(tf.shape(input_tensor)[0],), minval=-np.pi / 12, maxval=np.pi / 12)

    rotated_images = tf.contrib.image.transform(
        input_tensor,
        tf.contrib.image.angles_to_projective_transforms(
            random_angles, tf.cast(tf.shape(input_tensor)[1], tf.float32), tf.cast(tf.shape(input_tensor)[2], tf.float32)
        ))

    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda: rotated_images, lambda: input_tensor)
#

def input_enhance(input_tensor): # random noise
    max_pixel = 5 / 127.5 - 1.0
    min_pixel = -5 / 127.5 - 1.0
    random_pixel = tf.random.uniform(shape=tf.shape(input_tensor),minval = min_pixel , maxval = max_pixel)
    random_pixel = random_pixel/127.5-1
    pixel_image = input_tensor + random_pixel
    pixel_image = tf.clip_by_value(pixel_image, -1, 1)

    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda:pixel_image , lambda: input_tensor,tf.float32)
#
def random_flip(images):
    prob =  randint(0,1)
    for i in range(images.shape[0]):
        if prob==1:
            images[i] = images.transpose(Image.FLIP_LEFT_RIGHT)

    return images

def input_diversity(input_tensor): # 数据预处理操作


    input_tensor = input_enhance(input_tensor) # random noise
    input_tensor = input_scaled(input_tensor) # resize and padding
    input_tensor = input_rotate(input_tensor) # random rotation
    # todo
    # input_tensor = random_flip(input_tensor) #random flip?



    return tf.reshape(input_tensor, [-1, 112 ,112 ,3])
