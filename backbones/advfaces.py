import sys
import time

from imageio import imread
import math
from numpy import cov
import imp
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from functools import partial

from scipy import linalg
import warnings

from utils import tfutils
from scipy.linalg import sqrtm
from numpy import iscomplexobj
from numpy import trace


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_activations(images, sess, batch_size=50, verbose=False):
    # batch_size=50
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    n_images = images.shape[0]
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    n_batches = n_images // batch_size  # drops the last batch if < batch_size

    pred_arr = np.empty((n_batches * batch_size, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size

        if start + batch_size < n_images:
            end = start + batch_size
        else:
            end = n_images

        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch.shape[0], -1)
    if verbose:
        print(" done")
    return pred_arr
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
              #shape = [s.value for s in shape] TF 1.x
              shape = [s for s in shape] #TF 2.x
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3
def factor_function(input):
    x1 = tf.cast(input, dtype=tf.float32)
    factor1 = tf.add(1.0,tf.exp(tf.multiply(5.0,x1)))  # 1+e^10x
    factor1 = tf.divide(1.0,factor1) # 1/1+e^10x
    factor2 = tf.multiply(6.0,factor1) # 6/1+e^10x

    factor = tf.add(10.0,factor2) # 10+ 6/1+e^10x
    return factor

class AdvFaces:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True,
            log_device_placement=False,
        )
        self.sess = tf.Session(graph=self.graph, config=tf_config)

    def initialize(self, config, num_classes):
        """
            Initialize the graph from scratch according config.
        """
        with self.graph.as_default():
            with self.sess.as_default():
                G_grad_splits = []
                D_grad_splits = []
                average_dict = {}
                concat_dict = {}

                def insert_dict(_dict, k, v):
                    if k in _dict:
                        _dict[k].append(v)
                    else:
                        _dict[k] = [v]

                # Set up placeholders
                h, w = config.image_size
                channels = config.channels
                self.disc_counter = config.disc_counter

                self.mode = config.mode

                self.aux_matcher = imp.load_source("network_model",
                    config.aux_matcher_definition)

                summaries = []

                self.images = tf.placeholder(
                    tf.float32, shape=[None, h, w, channels], name="images"
                )
                self.t = tf.placeholder(tf.float32, shape=[None, h, w, channels])
                self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
                self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
                self.phase_train = tf.placeholder(tf.bool, name="phase_train")
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_step")

                self.setup_network_model(config, num_classes)

                if self.mode == "target":
                    self.perturb, self.G = self.generator(self.images, self.t)
                else:
                    self.perturb, self.G = self.generator(self.images)

                ########################## FID LOSS ###########################
                # with tf.Session() as sess:
                #     sess.run(tf.global_variables_initializer())
                #     m1, s1 = calculate_activation_statistics(self.images, sess)
                #     m2, s2 = calculate_activation_statistics(self.G, sess)
                #     fid_value = calculate_frechet_distance(m1, s1, m2, s2)

                ########################## GAN LOSS ###########################
                self.D_real = self.discriminator(self.images)
                self.D_fake = self.discriminator(self.G)
                d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.D_real, labels=tf.ones_like(self.D_real)
                    )
                )
                d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.D_fake, labels=tf.zeros_like(self.D_fake)
                    )
                )
                g_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake))
                )
                self.d_loss = d_loss_real + d_loss_fake

                ########################## IDENTITY LOSS #######################
                with slim.arg_scope(inception_arg_scope()):
                    #网络生成的对抗样本的特征
                        self.fake_feat, _ = self.aux_matcher.inference(
                            self.G,
                            bottleneck_layer_size=512,
                            phase_train=False,
                            keep_probability=1.0,
                        )
                    #定向攻击目标的特征
                        if self.mode == "target":
                            self.real_feat, _ = self.aux_matcher.inference(
                                self.t,
                                bottleneck_layer_size=512,
                                phase_train=False,
                                keep_probability=1.0,
                                reuse=True,
                            )
                            #输入图像本身的特征
                        else:
                            self.real_feat, _ = self.aux_matcher.inference(
                                self.images,
                                bottleneck_layer_size=512,
                                phase_train=False,
                                keep_probability=1.0,
                                reuse=True,
                            )
                if self.mode == "target": # cosine_pair 越大越好
                    identity_loss = tf.reduce_mean(1.0 - (tfutils.cosine_pair(self.fake_feat, self.real_feat) + 1.0)/ 2.0)
                else:# 非定向攻击时与同类的图像之间的cosine pair越小越好
                    identity_loss = tf.reduce_mean(tfutils.cosine_pair(self.fake_feat, self.real_feat))  #  计算张量的各个维度上的元素的平均值
                # 1.lfunction loss augment (x)
                # delta_identity_loss_factor =factor_function(identity_loss)
                # print("identity loss :", identity_loss,"identity loss factor:", delta_identity_loss_factor)


                 # 2.******identity cosine loss augment****
                # identity_loss = tf.cond(tf.less(identity_loss, 0), lambda: tf.multiply(identity_loss, 1.4641),
                #                         lambda: tf.multiply(identity_loss, 1.331))  #
                # identity_loss = tf.cond(tf.less(identity_loss, -0.2), lambda: tf.multiply(identity_loss, 1.05),
                #                         lambda: tf.multiply(identity_loss, 1))  #
                # identity_loss = tf.cond(tf.less(identity_loss, -0.4), lambda: tf.multiply(identity_loss, 1.05),
                #                         lambda: tf.multiply(identity_loss, 1))  #
                # # identity_loss = tf.cond(tf.less(identity_loss, -0.6), lambda: tf.multiply(identity_loss, 1.05),
                # #                         lambda: tf.multiply(identity_loss, 1))  #
                # identity_loss = tf.cond(tf.greater(identity_loss, 0.2), lambda: tf.multiply(identity_loss, 0.9090),
                #                         lambda: tf.multiply(identity_loss, 1))  #
                # identity_loss = tf.cond(tf.greater(identity_loss, 0.4), lambda: tf.multiply(identity_loss, 0.9090),
                #                         lambda: tf.multiply(identity_loss, 1))  #
                # identity_loss = tf.cond(tf.greater(identity_loss, 0.6), lambda: tf.multiply(identity_loss, 0.9090),
                #                         lambda: tf.multiply(identity_loss, 1))  #
                identity_loss = config.idt_loss_factor * identity_loss

                ########################## PERTURBATION LOSS #####################
                perturb_loss = config.perturb_loss_factor * tf.reduce_mean(
                                tf.maximum(tf.zeros((tf.shape(self.perturb)[0])) + config.MAX_PERTURBATION ,
                                tf.norm(tf.reshape( self.perturb, (tf.shape(self.perturb)[0], -1)),axis=1)  # 默认为L2距离
                                           )
                                           )

                ########################## PIXEL LOSS ############################
                pixel_loss = 1000.0 * tf.reduce_mean(tf.abs(self.G - self.images))

                self.total_loss = g_adv_loss + identity_loss + perturb_loss
                # gan_adv_loss:1  identity_loss:10  perturb_loss:1

                ################### LOSS SUMMARY ###################

                insert_dict(average_dict, "total_loss", self.total_loss)
                insert_dict(average_dict, "D_loss", self.d_loss)
                insert_dict(average_dict, "GAN_adv_loss", g_adv_loss)
                insert_dict(average_dict, "Identity_loss", identity_loss)
                # insert_dict(average_dict, "Identity_loss factor", delta_identity_loss_factor)
                insert_dict(average_dict, "Perturb_loss", perturb_loss)
                insert_dict(average_dict, "pixel_loss", pixel_loss)

                ################# VARIABLES TO UPDATE #################
                G_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator"
                )
                D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
                # learningrate decay
                # global_step = tf.Variable(0)
                # learning_rate =tf.train.exponential_decay(self.learning_rate, global_step, 50, 0.9) # decay

                self.train_G_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.total_loss, var_list=G_vars)  # Adam  WGAN-GP
                self.train_D_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=D_vars)

                for k, v in average_dict.items():
                    v = tfutils.average_tensors(v)
                    average_dict[k] = v
                    tfutils.insert(k, v)
                    if "loss" in k:
                        summaries.append(tf.summary.scalar("losses/" + k, v))
                    elif "acc" in k:
                        summaries.append(tf.summary.scalar("acc/" + k, v))
                    else:
                        tf.summary(k, v)
                for k, v in concat_dict.items():
                    v = tf.concat(v, axis=0, name="merged_" + k)
                    concat_dict[k] = v
                    tfutils.insert(k, v)
                trainable_variables = [t for t in tf.trainable_variables()]

                fn = [var for var in tf.trainable_variables() if config.aux_matcher_scope in var.name]
                print(trainable_variables)

                self.update_global_step_op = tf.assign_add(self.global_step, 1)
                summaries.append(tf.summary.scalar("learning_rate", self.learning_rate))
                self.summary_op = tf.summary.merge(summaries)

                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(trainable_variables, max_to_keep=None)
                f_saver = tf.train.Saver(fn)
                f_saver.restore(self.sess, config.aux_matcher_path)

                self.watch_list = tfutils.get_watchlist()

    def setup_network_model(self, config, num_classes):
        network_models = imp.load_source("network_model", config.network)
        self.generator = partial(
            network_models.generator,
            keep_prob=self.keep_prob,
            phase_train=self.phase_train,
            weight_decay=config.weight_decay,
            reuse=tf.AUTO_REUSE,
            scope="Generator",
        )

        self.generator_mask = partial(
            network_models.generator,
            keep_prob=self.keep_prob,
            phase_train=self.phase_train,
            weight_decay=config.weight_decay,
            reuse=tf.AUTO_REUSE,
            scope="Generator",
        )

        self.discriminator = partial(
            network_models.normal_discriminator,
            keep_prob=self.keep_prob,
            phase_train=self.phase_train,
            weight_decay=config.weight_decay,
            reuse=tf.AUTO_REUSE,
            scope="Discriminator",
        )
        

    def train(
        self,
        image_batch,
        target_batch,
        label_batch,
        learning_rate,
        num_classes,
        keep_prob,
    ):
        h, w, c = image_batch.shape[1:]
        feed_dict = {
            self.images: image_batch,
            self.learning_rate: learning_rate,
            self.keep_prob: keep_prob,
            self.t: target_batch,
            self.phase_train: True,
        }
        for i in range(1):
            _ = self.sess.run(self.train_G_op, feed_dict=feed_dict)

        _, wl, sm, step = self.sess.run(
            [
                self.train_D_op,
                tfutils.get_watchlist(),
                self.summary_op,
                self.update_global_step_op,
            ],
            feed_dict=feed_dict,
        )
        return wl, sm, step

    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES
        )
        tfutils.restore_model(self.sess, trainable_variables, *args, **kwargs)

    def save_model(self, model_dir, global_step):
        tfutils.save_model(self.sess, self.saver, model_dir, global_step)

    def decode_images(self, features, batch_size):
        num_images = features.shape[0]
        h, w, c = tuple(self.G.shape[1:])
        result = np.ndarray((num_images, h, w, c), dtype=np.float32)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            feat = features[start_idx:end_idx]
            feed_dict = {
                self.feats: feat,
                self.phase_train: False,
                self.keep_prob: 1.0,
            }
            result[start_idx:end_idx] = self.sess.run(self.G, feed_dict=feed_dict)
        return result

    def generate_images(self, images, targets=None, batch_size=128, return_targets=False):
        num_images = images.shape[0]
        h, w, c = tuple(self.G.shape[1:])
        # h, w, c = tuple(images.shape[1:])
        # print("h:{} w:{} c:{}".format(h,w,c))
        result = np.ndarray((num_images, h, w, c), dtype=np.float32)
        perturb = np.ndarray((num_images, h, w, c), dtype=np.float32)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            im = images[start_idx:end_idx]
            if self.mode == "target":
                t = targets[start_idx:end_idx]
                feed_dict = {
                    self.images: im,
                    self.t: t,
                    self.phase_train: False,
                    self.keep_prob: 1.0,  }
            else: # 非定向攻击
                feed_dict = {
                    self.images: im,
                    self.phase_train: False,
                    self.keep_prob: 1.0, }
            result[start_idx:end_idx], perturb[start_idx:end_idx] = self.sess.run(  [self.G, self.perturb], feed_dict=feed_dict )

        return result, perturb

    def aux_matcher_extract_feature(self, images, batch_size=512, verbose=True):
        num_images = images.shape[0]
        fake = np.ndarray((num_images, 512), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            im = images[start_idx:end_idx]
            if self.mode == 'target':
                feed_dict = {
                    self.t: im,
                    self.phase_train: False,
                    self.keep_prob: 1.0,
                }
            else:
                feed_dict = {
                        self.images: im,
                        self.phase_train: False,
                        self.keep_prob: 1.0,
                }
            fake[start_idx:end_idx] = self.sess.run(self.real_feat, feed_dict=feed_dict)
        return fake

    def load_model(self, *args, **kwargs):
        print("load_model")
        tfutils.load_model(self.sess, *args, **kwargs)
        self.phase_train = self.graph.get_tensor_by_name("phase_train:0")
        self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
        self.perturb = self.graph.get_tensor_by_name("Generator_1/output:0")
        self.G = self.graph.get_tensor_by_name("Generator_1/sub:0")
        self.D = self.graph.get_tensor_by_name("Discriminator/Reshape:0")
        self.images = self.graph.get_tensor_by_name("images:0")
        self.mode = "obfuscation" # 是否应该备注
        if self.mode == "target":
            self.t = self.graph.get_tensor_by_name("Placeholder:0")

###############################################################################
####################  ONLY NEEDED FOR FACENET MATCHER #########################
def inception_arg_scope(
    weight_decay=0.00004,
    use_batch_norm=True,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
):

    """Defines the obfuscation_attack_Advface_facenet arg scope for inception models.

  Args:

    weight_decay: The weight decay to use for regularizing the model.

    use_batch_norm: "If `True`, batch_norm is applied after each convolution.

    batch_norm_decay: Decay for batch norm moving average.

    batch_norm_epsilon: Small float added to variance to avoid dividing by zero

      in batch norm.

  Returns:

    An `arg_scope` to use for the inception models.

  """

    batch_norm_params = {
        # Decay for the moving averages.
        "decay": batch_norm_decay,
        # epsilon to prevent 0s in variance.
        "epsilon": batch_norm_epsilon,
        # collection containing update_ops.
        "updates_collections": tf.GraphKeys.UPDATE_OPS,
    }

    if use_batch_norm:

        normalizer_fn = slim.batch_norm

        normalizer_params = batch_norm_params

    else:

        normalizer_fn = None

        normalizer_params = {}

    # Set weight_decay for weights in Conv and FC layers.

    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
    ):

        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
        ) as sc:

            return sc
