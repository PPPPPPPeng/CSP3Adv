from backbones.insightface import *
from utils.util import *
from utils.attack import *
from backbones.unet import unet
from backbones.MobileFaceNet import mobilefacenet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import cv2
import tensorflow as tf

import time



def mobile(inputs): # 使用MobileFace 对图像进行处理，得到特征向量
    prelogits, net_points = mobilefacenet(inputs, bottleneck_layer_size=192, reuse=tf.AUTO_REUSE)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    return embeddings

def test(args):
    epoch = args.train_epoch  # train_epoch = 50  config_ms1m_100：   20epochs * 100000
    cnt = 0
    total_time = 0
    print("total epoch:",epoch)
    batch_size = args.train_batchsize # config_ms1m_100: 100
    config = yaml.load(open(args.config_path),Loader=yaml.FullLoader) #加载参数

    benchmark = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_benchmark') #probe
    images = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_image') # 加载图像
    # train_placeholder = tf.placeholder(tf.bool)
    # keep_props = tf.placeholder(tf.float32)

    # 得到MobileFace处理后的特征向量
    benchmark_embds = mobile(benchmark)
    embds = mobile(images)
    # 得到Insightface/arcface处理后的特征向量
    arc_embds, _ = get_embd(images, config)
    arc_ben_embds, _ = get_embd(benchmark, config)

    # 默认计算和MobileFace-benchmark之间的特征距离
    def get_distance(embds1, embds2=benchmark_embds,metric =0):
        if metric == 0:  # Square Euclidian distance
            embeddings1 = embds1 / tf.norm(embds1, axis=1, keepdims=True)
            embeddings2 = embds2 / tf.norm(embds2, axis=1, keepdims=True)
            diff = tf.subtract(embeddings1, embeddings2)
            distance = tf.reduce_sum(tf.multiply(diff, diff), axis=1)  # 按行求和
            return distance
        if metric == 1:
            # Distance based on
            dot = tf.reduce_sum(tf.multiply(embds1, embds2), axis=1)
            norm = tf.norm(embds1, axis=1) * tf.norm(embds2, axis=1)
            similarity = dot / norm  # cosine similarity
            dist = tf.acos(similarity) / math.pi  # cosine distance
            return dist

    # 用户模型进行攻击，获取梯度方向
    # grad_op = tf.gradients(dist, inputs)[0]
    # dist = get_distance(embds)
    # x_FGM = FGM(images, dist)
    # x_FGSM= FGSM(images, dist) # ok
    # x_IFGSM = IFGSM(images, lambda f_embd: mobile(f_embd),lambda f_dis: get_distance(f_dis), 1) #
    # x_TIFGSM = TIFGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis), 1)  # ok
    # x_MIFGSM = MIFGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis),1) #
    x_DI2FGSM = DI2FGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis), 1) #
    # x_MDI2FGSM = MDI2FGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis),1) #


    adv = x_DI2FGSM
    x_noise = adv - images

    # 云模型处理
    distances = get_distance(arc_embds, arc_ben_embds, 0) # Eculidean or cosine distance
    # threshold = 1.12
    # threshold = 1.16 #agedb30
    threshold = 1.29 #cfpfp
    # threshold = 0.355
    distances = threshold - distances
    prediction = tf.sign(distances) # distances距离小于threshold即二者属于同一人，prediction = 0,二者为同一人, or else prediction = -1
    correct_prediction = tf.count_nonzero(prediction + 1, dtype=tf.float32)
    # 计算original ASR
    accuracy = correct_prediction/batch_size # 二者为同一人的概率

    # 2.对抗梯度迁移
    output= unet(x_noise)
    loss_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='APF')

    eps = 8 / 255. * 2  # maximum perturbation of each pixel
    print('eps = ',eps)
    s = tf.clip_by_value(output, - eps, eps)
    image_adv = tf.clip_by_value(images + s, -1.0, 1.0)

    # accuracy = accurate(x, y)
    variables_unet = tf.contrib.framework.get_variables_to_restore(include=['APF'])
    saver_unet = tf.train.Saver(variables_unet)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    test_loss = 0
    # X = create_lfw_npy()  # 读取LFW测试集图像，创建6000张测试集图片
    # X = create_agedb_npy()  # 读取AGEDB-30测试集图像，创建6000张测试集图片
    X = create_cfp_npy()    # 读取CFP-FP测试集图像，创建7000张测试集图片
    #### If you wanna test your own data, please use the following code:###
    #                                                                     #
    #    X = np.load(args.test_data)                                      #
    #    X = X / 127.5 - 1.0                                              #
    #                                                                     #
    #######################################################################
    print('测试集图片数目，图像大小:', X.shape)
    X_0 = X[0::2]  # 切片，从第0张开始每隔两张图片取出一张图片
    X_1 = X[1::2]  # 切片，从第1张开始每隔两张图片取出一张图片
    num_test = len(X_0)
    print('测试集对象数目:', num_test, 'batch_size:', batch_size)

    variables_arc = tf.contrib.framework.get_variables_to_restore(include=['embd_extractor'])
    variables_mobilefacenet = tf.contrib.framework.get_variables_to_restore(include=['MobileFaceNet'])
    saver_m = tf.train.Saver(variables_mobilefacenet)
    saver_a = tf.train.Saver(variables_arc)

    start = time.clock()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver_m.restore(sess, args.mobilefacenet_model_path)  # the path you save the Mobilefacenet model.
        saver_unet.restore(sess, args.test_model )
        saver_a.restore(sess, args.insightface_model_path)

        for j in range(num_test//batch_size): # 分batch进行计算
            x = X_0[j*batch_size:(j+1)*batch_size] # batch input images
            ben = X_1[j * batch_size:(j + 1) * batch_size] # batch probe images
            # AE
            x_adv = sess.run(image_adv, feed_dict={images:x, benchmark:ben})
            user_adv = sess.run(adv, feed_dict={images:x, benchmark:ben})

            # calculate single pictures quality
            for i in range(batch_size):
                # X 已经是np.array格式了
                print("Batch: {}/{} - pic: {}/{}".format(j+1,num_test//batch_size,i+1,batch_size))
                misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/DI2FGSM-AE/{}.jpg'.format(cnt),x_adv)
                cnt += 1
        end = time.clock()
        total_time += end - start
        print("-------------------")
        mean_time = total_time / num_test
        print("Computation Time (s): {}".format(mean_time))



