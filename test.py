from backbones.insightface import *
from utils.util import *
from utils.attack import *
from backbones.unet import unet
from backbones.MobileFaceNet import mobilefacenet
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import cv2
import tensorflow as tf
import random
import time


def mobile(inputs): # 使用MobileFace 对图像进行处理，得到特征向量
    prelogits, net_points = mobilefacenet(inputs, bottleneck_layer_size=192, reuse=tf.AUTO_REUSE)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    return embeddings

def test(args):
    epoch = args.train_epoch  # train_epoch = 50  config_ms1m_100：   20epochs * 100000
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
    # x_TDI2FGSM = TDI2FGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis), 1)  #
    # x_MDI2FGSM = MDI2FGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis),1) #


    adv = x_DI2FGSM
    x_noise = adv - images

    # 云模型处理
    distances = get_distance(arc_embds, arc_ben_embds, 0) # Eculidean or cosine distance
    # threshold = 1.12 # LFW
    # threshold = 1.1 # CALFW
    # threshold = 1.13 # CPLFW
    # threshold = 1.16 # agedb30
    threshold = 1.29 # cfpfp
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
    image_adv = tf.clip_by_value(images + s, -1.0, 1.0) # 测试阶段不采用高斯滤波

    # accuracy = accurate(x, y)
    variables_unet = tf.contrib.framework.get_variables_to_restore(include=['APF'])
    saver_unet = tf.train.Saver(variables_unet)
    total_time = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    wo_acc = 0
    ori_acc = 0
    user_acc = 0
    user_noise_acc = 0
    ss = []
    ss_user = []
    ps = []
    ps_user = []
    test_loss = 0
    cnt = 0
    # X = create_lfw_npy_samepair()  # 读取LFW测试集图像，创建6000张测试集图片
    # X = create_calfw_npy()  # 读取CALFW测试集图像，创建6000张测试集图片
    # X = create_cplfw_npy()  # 读取CPLFW测试集图像，创建6000张测试集图片
    # X = create_agedb_npy()  # 读取AGEDB-30测试集图像，创建6000张测试集图片
    X = create_cfp_npy()     # 读取CFP-FP测试集图像，创建7000张测试集图片
    #### If you wanna test your own data, please use the following code:###
    #                                                                     #
    #    X = np.load(args.test_data)                                      #
    #    X = X / 127.5 - 1.0
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

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver_m.restore(sess, args.mobilefacenet_model_path)  # the path you save the Mobilefacenet model.
        saver_unet.restore(sess, args.test_model )
        saver_a.restore(sess, args.insightface_model_path)
        saveplace = os.path.split(args.test_model)[0]

        for j in range(num_test//batch_size): # 30个batch进行计算
            x = X_0[j*batch_size:(j+1)*batch_size] # batch input images
            ben = X_1[j * batch_size:(j + 1) * batch_size] # batch probe images
            # AE
            start = time.clock()
            x_adv = sess.run(image_adv, feed_dict={images:x, benchmark:ben})
            end = time.clock()
            total_time += end - start
            user_adv = sess.run(adv, feed_dict={images:x, benchmark:ben})
            user_noise = user_adv - x

            # ACC
            ori_acc += sess.run(accuracy, feed_dict={images: x, benchmark: ben})
            user_acc += sess.run(accuracy, feed_dict={images: user_adv, benchmark: ben})
            user_noise_acc += sess.run(accuracy, feed_dict={images: user_noise, benchmark: ben})




            # calculate single pictures quality　　batchsize=100
            for i in range(batch_size): #图像质量评价处理
                # X 已经是np.array格式了
                print("Batch: {}/{} - pic: {}/{}".format(j+1,num_test//batch_size,i+1,batch_size))
                # misc.imsave(saveplace+'/calfwAE/{}.png'.format(cnt), x_adv[i])
                misc.imsave(saveplace + '/CFPnoise/{}.png'.format(cnt), user_noise[i])
                # misc.imsave(saveplace + '/userAE-cfp/{}.png'.format(cnt), user_adv[i])
                # misc.imsave(saveplace+'/Noise/{}.png'.format(cnt), (user_adv[i] - x[i]))
                # misc.imsave(saveplace + '/endNoise/{}.png'.format(cnt), (x_adv[i] - x[i]))

                # JPEG
                # temp = Image.open(saveplace+'/calfwAE/{}.png'.format(cnt)).convert('RGB')
                # temp.save(saveplace+'/JPEGlfwAE/{}.jpeg'.format(cnt),format='JPEG',quality=30)
                # x_adv[i] = misc.imread(saveplace + '/JPEGlfwAE/{}.jpeg'.format(cnt))

                # webp
                # t = cv2.imread(saveplace+'/calfwAE/{}.png'.format(cnt))
                # cv2.imwrite(saveplace + '/WEBPlfwAE/{}.webp'.format(cnt),t,[cv2.IMWRITE_WEBP_QUALITY,30])
                # x_adv[i] = misc.imread(saveplace + '/WEBPlfwAE/{}.webp'.format(cnt))

                #jpeg2000 /

                # Compression  operation
                # x_adv[i] = x_adv[i] / 127.5 - 1.0
                # os.remove(saveplace + '/JPEGlfwAE/{}.jpeg'.format(cnt)) # todo
                # os.remove(saveplace + '/WEBPlfwAE/{}.webp'.format(cnt))  # todo

                cnt += 1

                # LPIPS
                # x[i]=np.round((x[i]+1) * 127.5).astype(np.uint8)
                # x_adv[i] = np.round((x_adv[i] + 1) * 127.5).astype(np.uint8)
                #
                # # Lpips
                # image0_ph = tf.placeholder(tf.float32)
                # image1_ph = tf.placeholder(tf.float32)
                # distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')
                # distance = sess.run(distance_t,
                #                     feed_dict={image0_ph: x[i].astype(np.uint8), image1_ph: x_adv[i].astype(np.uint8)})
                # print(" Lpips: {}".format(distance))
                # results.append(distance)
                # tf.get_default_graph().finalize()

                # SSIM\PSNR
                im1 = cv2.cvtColor(x[i].astype('float32'), cv2.COLOR_BGR2GRAY)  # batch input images
                im2 = cv2.cvtColor(x_adv[i].astype('float32'), cv2.COLOR_BGR2GRAY)  # batch AE probe images
                im3 = cv2.cvtColor(user_adv[i].astype('float32'), cv2.COLOR_BGR2GRAY)  # batch user probe images
                # 最终AE的图像质量评价
                s_temp = ssim(im1, im2)
                if(s_temp!= 1):
                    ss.append(s_temp)
                p_temp = psnr(im1, im2)
                if(p_temp<500):
                    ps.append(p_temp)
                print("SSIM: {} PSNR: {} ".format(s_temp, p_temp))

                s_t = ssim(im1,im3)
                if (s_t != 1):
                    ss_user.append(s_t)
                p_t = psnr(im1, im3)
                if (p_t < 500):
                    ps_user.append(p_t)
            # JPEG后的Acc
            wo_acc += sess.run(accuracy, feed_dict={images: x_adv, benchmark: ben})  # 计算对抗攻击后的识别准确率
            s = np.mean(ss)
            s_stand = np.std(ss)
            p = np.mean(ps)
            p_stand = np.std(ps)

            s_user = np.mean(ss_user)
            s_user_stand = np.std(ss_user)
            p_user = np.mean(ps_user)
            p_user_stand = np.std(ps_user)
            print("User AE    : {:.2}±{:.2}/{:.4}±{:.2} ".format(s_user, s_user_stand, p_user, p_user_stand))
            print("最终  AE   : {:.2}±{:.2}/{:.4}±{:.2} ".format(s, s_stand, p, p_stand))
            mean_time = total_time / num_test
            print("Average Computation Time (s): {}".format(mean_time))
        print("-------------------")
        print('Insightface Ori acc={:.5}'.format(ori_acc/(num_test//batch_size)))
        print('Insightface User-NOISE acc{:.5} ASR ={:.4} '.format(1-user_noise_acc/(num_test//batch_size), (user_noise_acc/(num_test//batch_size))*100))
        print('Insightface AE acc={:.5} ASR ={:.4} '.format(wo_acc/(num_test//batch_size),(1-wo_acc/(num_test//batch_size))*100  ))



