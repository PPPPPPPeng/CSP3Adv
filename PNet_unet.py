from backbones.insightface import * # insightface-get_embd()
from utils.util import *
from utils.attack import *
from backbones.unet import unet
from backbones.MobileFaceNet import mobilefacenet
from utils.mmd import *
import random
from backbones.inception_resnet_v1 import inception_resnet_v1 # FaceNet
import sphere_torch   #SphereFace
import os
from advfaces import AdvFaces
from utils.imageprocessing import preprocess
from torchvision import utils as vutils
import datetime
import utils.differentiableJPEG
from skimage.restoration import denoise_tv_bregman as _denoise_tv_bregman
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# distance ??
#   insightface.get_embd : 黑盒测试获取特征向量
# 在Insightface模型上测试黑盒ASR
# 通用扰动加强E在UNET.PY

kernel = gkern(3, 7.9).astype(np.float32)  # sigma=(k-1)/6 ？
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)
def median_filter(data, filt_length=3):
    '''
    Computes a median filtered output of each [n_bins, n_channels] data sample
        and returns output w/ same shape, but median filtered

    NOTE: as TF doesnt have a median filter implemented, this had to be done in very hacky way...
    '''
    edges = filt_length// 2

    # convert to 4D, where data is in 3rd dim (e.g. data[0,0,:,0]
    exp_data = tf.expand_dims(tf.expand_dims(data, 0), -1)
    # get rolling window
    wins = tf.image.extract_patches(images=exp_data, sizes=[1, filt_length, 1, 1],
                       strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    # get median of each window
    wins = tf.math.top_k(wins, k=2)[0][0, :, :, edges]
    # Concat edges
    out = tf.concat((data[:edges, :], wins, data[-edges:, :]), 0)

    return out

def mobile(inputs):
    prelogits, net_points = mobilefacenet(inputs, bottleneck_layer_size=192, reuse=tf.AUTO_REUSE)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    return embeddings

def jpeg_compress(x, quality=75):
    return tf.image.decode_jpeg(tf.image.encode_jpeg(x, format='rgb', quality=quality),channels=3)

def slq(x, qualities=(20, 40, 60, 80), patch_size=8):
    print("begin slq")
    num_qualities = len(qualities)

    with tf.name_scope('slq'):
        one = tf.constant(1, name='one')
        zero = tf.constant(0, name='zero')

        x_shape = tf.shape(x)
        n, m = x_shape[0], x_shape[1]
        patch_n = tf.cast(n / patch_size, dtype=tf.int32) \
                  + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)
        patch_m = tf.cast(m / patch_size, dtype=tf.int32) \
                  + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)

        R = tf.tile(tf.reshape(tf.range(n), (n, 1)), [1, m])
        C = tf.reshape(tf.tile(tf.range(m), [n]), (n, m))
        Z = tf.image.resize_nearest_neighbor(
            [tf.random_uniform((patch_n, patch_m, 3),0, num_qualities, dtype=tf.int32)],
            (patch_n * patch_size, patch_m * patch_size),
            name='random_layer_indices')[0, :, :, 0][:n, :m]
        indices = tf.transpose(
            tf.stack([Z, R, C]),
            perm=[1, 2, 0],
            name='random_layer_indices')

        x_compressed_stack = tf.stack(list(map(lambda q: tf.image.decode_jpeg(tf.image.encode_jpeg(x, format='rgb', quality=q), channels=3),qualities)),name='compressed_images')

        x_slq = tf.gather_nd(x_compressed_stack, indices, name='final_image')

    return x_slq


def denoise_tv_bregman(img_arr, weight=30):
    denoised = _denoise_tv_bregman(img_arr, weight=weight) * 255.
    return np.array(denoised, dtype=img_arr.dtype)

def train(args):
    # args = get_args()
    defense_name = 'jpeg'# only 1 defense
    defenses = ['jpeg', 'slq', 'median_filter', 'tv_bregman']
    tf_defenses = ['jpeg', 'slq',]
    defense_options = {
        'jpeg': {'quality': 80},
        'slq': {},
        'median_filter': {},
        'tv_bregman': {'weight': 30}}
    defense_options_updated = defense_options[defense_name]
    defense_fn_map = {
        'jpeg': jpeg_compress,
        'slq': slq,
        'median_filter': median_filter,
        'tv_bregman': denoise_tv_bregman}
    defense_fn = defense_fn_map[defense_name]
    epoch = args.train_epoch # 50
    print("total epoch:",epoch)
    batch_size = args.train_batchsize
    config = yaml.load(open(args.config_path),Loader=yaml.FullLoader)

    benchmark = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_benchmark')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_image')


    benchmark_embds = mobile(benchmark) # user-end benchmark
    embds = mobile(images) # user-end input embeddings

    def get_distance(embds1, embds2=benchmark_embds,metric = 0):# 计算L1距离
        if metric == 0:  # Square Euclidian distance
            embeddings1 = embds1 / tf.norm(embds1, axis=1, keepdims=True)
            embeddings2 = embds2 / tf.norm(embds2, axis=1, keepdims=True)
            diff = tf.subtract(embeddings1, embeddings2)
            distance = tf.reduce_sum(tf.multiply(diff, diff), axis=1) # 按行求和
            return distance
        if metric == 1:
            # Distance based on coisne distance
            # dot = tf.reduce_sum(tf.multiply(embds1, embds2), axis=1)
            # norm = tf.norm(embds1, axis=1) * tf.norm(embds2, axis=1)
            # similarity = dot / norm  # cosine similarity
            # dist = tf.acos(similarity) / math.pi  # cosine distance
            # return dist

            # cos = Xi * Yi / (|| Xi || * || Yi || )

            epsilon = 1e-10
            x1_norm = tf.sqrt(tf.reduce_sum(tf.square(embds1), axis=1, keepdims=True))
            #
            x2_norm = tf.sqrt(tf.reduce_sum(tf.square(embds2), axis=1, keepdims=True))
            x1 = embds1 / (x1_norm + epsilon)
            x2 = embds2 / (x2_norm + epsilon)
            dist = tf.reduce_sum(x1 * x2, axis=1)
            return dist







    # user adversarial attack
    # grad_op = tf.gradients(dist, inputs)[0]
    # dist = get_distance(embds)
    # x_FGM = FGM(images, dist)
    # x_FGSM= FGSM(images, dist) # ok
    # x_IFGSM = IFGSM(images, lambda f_embd: mobile(f_embd),lambda f_dis: get_distance(f_dis), 1) # ok
    # x_PGD = PGD(images, lambda f_embd: mobile(f_embd),lambda f_dis: get_distance(f_dis), 1)
    # x_TIFGSM = TIFGSM(images, lambda f_embd: mobile(f_embd),lambda f_dis: get_distance(f_dis), 1) # ok
    # x_MIFGSM = MIFGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis),1) # ok
    x_DI2FGSM = DI2FGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis), 1) # ok
    # x_MDI2FGSM = MDI2FGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis),1) # ok

# TAdvFace
#     network = AdvFaces()
#     network.load_model('/home/yl/PycharmProjects/sumail/AdvFace/log/obfuscation_attack_Advface_facenet/eps=16')  # 本地训练的模型 need to modify
#     images = preprocess(images, config, is_training=False)
#     x_adv, _ = network.generate_images(images)

    x_adv = x_DI2FGSM # user-end adv
    x_noise = x_adv - images # x_noise :用户端最终计算得到的 添加的噪声

    # insightface/arcface 112*112
    arc_embds, _ = get_embd(images, config)  # 使用insightface提取 原始图片/对抗样本 特征向量,用于计算Acc
    arc_ben_embds, _ = get_embd(benchmark, config)  # 使用insightface提取probe特征向量，用于计算Acc


    #  2. Adversarial Gradient Transfer T(')
    print("input shape:",x_noise.shape)
    output = unet(x_noise) # 为了获得更好的黑盒迁移性，集成Facenet,Sphereface模型进行梯度迁移计算。
    print("output shape:", output.shape)
    loss_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='APF')



    # clip to get the adv
    eps = 8 / 255. * 2
    s = tf.clip_by_value(output, - eps, eps) # clip (s)
    # before DEFENSE
    image_adv = tf.clip_by_value(images + s, -1.0, 1.0) # adv
    # with tf.Session() as sess:
    #     image_adv_NUMPY = image_adv.eval()
    # image_adv_NUMPY =utils.differentiableJPEG.jpeg_compress_decompress(image_adv_NUMPY,rounding=utils.differentiableJPEG.diff_round,factor=utils.differentiableJPEG.quality_to_factor(81))
    # image_adv = tf.convert_to_tensor(image_adv_NUMPY)
    # print(image_adv_NUMPY)
    image_adv = tf.nn.depthwise_conv2d(image_adv, stack_kernel, strides=[1, 1, 1, 1], padding='SAME') # Guassian filter 仅在训练阶段出现高斯滤波
    print("pass guassian kernel")
    # image_adv = image_adv + random.gauss(0,0.005) # guassian 97
    image_adv = tf.clip_by_value(image_adv, -1.0, 1.0)

    # after DEFENSE
    # image_adv = images + output  # 最终的AE，image_adv = x_adv
    # image_adv = tf.nn.depthwise_conv2d(image_adv, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    # s = image_adv-images
    # s = tf.clip_by_value(s, - eps, eps)  # clip (s)
    # image_adv = images + s
    # image_adv = tf.clip_by_value(image_adv, -1.0, 1.0)



    # defense Apply preprocessing for TF defenses 仅对生成的对抗样本压缩
    # if defense_name in tf_defenses:
    #     image_adv = tf.cast(image_adv, dtype=tf.uint8)
    # #     print("in tf_defenses")
    # #     print("defense fn:", defense_fn)
    #     image_adv = tf.map_fn(lambda x: defense_fn(x, **defense_options_updated), image_adv)
    # image_adv = tf.cast(image_adv, dtype=tf.float32)


    # cloud computing :
    # facenet 160*160 todo

    # images = tf.image.resize(images, [160, 160])
    # benchmark = tf.image.resize(benchmark, [160, 160])
    # facenet_embds, _ = inception_resnet_v1(images)
    # facenet_ben_embds, _ = inception_resnet_v1(benchmark)

    # sphereface-torch 112*96 todo

    # print(images)
    # images = images.eval(session=tf.Session())
    # print(type(images))
    # images = tf.image.resize(images, [112, 96])
    # sphereface = getattr(sphere_torch, 'sphere20a')()
    # state_dict = torch.load(args.sphereface_model_path)
    #
    # sphereface.load_state_dict(state_dict)
    # sphereface.cuda()
    # sphereface.eval()
    # sphereface.feature = True
    # sphereface_embds = sphereface(images)

    # 根据距离计算ACC  todo 改成集成
    distances = get_distance(arc_embds, arc_ben_embds) # 计算在Insightface下计算得到的 原始图片/对抗样本 与probe图片之间的特征距离
    # distances = get_distance(facenet_embds, facenet_ben_embds)

    # threshold = 1.02  # 官方的 arcface-for LFW
    threshold = 1.12  # 自己测出来的threshold arcface-for LFW
    # threshold = 1.16 # 自己测出来的threshold arcface-for Agedb_30
    # threshold = 1.12 # 自己测出来的threshold arcface-for CFP_FP
    # threshold = 0.355

    distances = threshold - distances
    prediction = tf.sign(distances)
    correct_prediction = tf.count_nonzero(prediction + 1, dtype=tf.float32)
    accuracy = correct_prediction / batch_size  # 【函数】计算识别准确率

    # 根据距离计算损失函数
    embds_adv, _ = get_embd(image_adv, config) # 提取最终AE的特征向量 todo
    embds_ben_arc, _ = get_embd(benchmark, config) # 提取Probe的特征向量 todo
    distances_adv_E = get_distance(embds_adv, embds_ben_arc)  # 计算最终AE与probe之间的距离
    # distances_adv_C = get_distance(embds_adv, embds_ben_arc,metric =1 ) # 计算的是特征向量之间的cosine similarity score

    # embds_adv_facenet, _ = inception_resnet_v1(image_adv)
    # embds_ben_facenet, _ = inception_resnet_v1(benchmark)
    # distances_adv_s = get_distance(embds_adv_facenet, embds_ben_facenet)


    # loss_distance_C =  2 * distances_adv_C  # cosine loss weight 0.1  0.5  1
    loss_distance_E = 5 - distances_adv_E

    loss_E = tf.reduce_mean(loss_distance_E)  # distance_loss
    # loss_C = tf.reduce_mean(loss_distance_C)

    # compute image-level natural loss by MMD.
    image_adv_temp = image_adv
    images_temp = images
    MMD = mmd_loss(tf.reshape(image_adv_temp,(batch_size, -1)), tf.reshape(images_temp,(batch_size, -1))) # todo

    image_user_temp = x_adv
    # weight = tf.placeholder(tf.float32)
    # loss_MMD = tf.multiply(0.2, MMD)

    # show L0
    l0_user_loss = tf.cast(tf.reduce_sum(tf.where(tf.greater(tf.abs(x_noise), 0))), dtype=tf.float32)
    l0_adv_loss = tf.cast(tf.reduce_sum(tf.where(tf.greater(tf.abs(output), 0))), dtype=tf.float32)

    # show l1
    l1_user_loss = tf.reduce_sum(tf.abs(x_noise))/50/112/112*100
    l1_adv_loss = tf.reduce_sum(tf.abs(output))/50/112/112*100

    # l2 loss
    l2_user_loss = tf.nn.l2_loss(x_noise)



    loss = loss_E# min loss
    optimizer = tf.train.AdamOptimizer(0.0001) # Lr
    train_op = optimizer.minimize(loss, var_list=loss_vars)  # 根据损失函数训练对抗梯度迁移模型的参数???
    # accuracy = accurate(x, y)

    variables_unet = tf.contrib.framework.get_variables_to_restore(include=['APF'])
    saver_unet = tf.train.Saver(variables_unet) # save model : T + E

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 读取MobileFace和Insightface的预训练权重
        sess.run(tf.global_variables_initializer())

        variables_mobilefacenet = tf.contrib.framework.get_variables_to_restore(include=['MobileFaceNet'])
        saver_m = tf.train.Saver(variables_mobilefacenet)

        variables_arc = tf.contrib.framework.get_variables_to_restore(include=['embd_extractor'])
        saver_a = tf.train.Saver(variables_arc)

        # variables_sphere = tf.contrib.framework.get_variables_to_restore(include=['sphere_network'])
        # saver_sphere = tf.train.Saver(variables_sphere)

        # variables_facenet = tf.contrib.framework.get_variables_to_restore(include=['InceptionResnetV1'])
        # saver_facenet = tf.train.Saver(variables_facenet)

        # the path you save the Mobilefacenet model.
        saver_m.restore(sess, args.mobilefacenet_model_path)
        saver_a.restore(sess, args.insightface_model_path)
        # saver_sphere.restore(sess, args.sphereface_model_path)
        # saver_facenet.restore(sess, args.facenet_model_path)

        # X = create_lfw_npy_train() #测试集图片
        X = create_lfw_npy_samepair()  # samepair测试集图片

        print(X.shape)
        X_0 = X[0::2]
        X_1 = X[1::2]
        # num_test = len(X_0)
        num_test = X_0.shape[0]
        print("num_test:",num_test)
        # the path you save the train data.
        train_dataset = np.load(args.train_data,allow_pickle=True) # 训练集合图片
        # print(train_dataset.shape)

        list_img = np.array(train_dataset) / 127.5 - 1.0
        list_img = np.reshape(list_img, [-1, 112, 112, 3])
        list_img_0 = list_img[0::2]
        list_img_1 = list_img[1::2]
        len_train = len(list_img_0)

        n_batch = len_train//batch_size
        print('训练数据集对象个数', len_train,'   batch_size = ',batch_size,'    共计batch数目 = ',n_batch)
        # saver = tf.train.Saver(max_to_keep=1)
        best_acc = 1.0
        best_train = 100
        for i in range(epoch):
            if i ==0 :print("begin training")
            for batch in range(n_batch): # 分batch采用训练集图片进行训练
                x_batch = list_img_0[batch*batch_size:(batch+1)*batch_size]
                y_batch = list_img_1[batch * batch_size:(batch + 1) * batch_size]


                sess.run(train_op,feed_dict={images: x_batch, benchmark: y_batch})  # training every batch

                if batch % 50 == 0: # 每50个batch计算当前的test loss等信息

                    train_loss = sess.run(loss, feed_dict={images:x_batch, benchmark:y_batch}) # loss

                    att_acc = 0
                    test_loss = 0
                    for j in range(num_test//batch_size): # 测试图片 分batch(100) 对图片进行test accuracy
                        x = X_0[j*batch_size:(j+1)*batch_size]
                        ben = X_1[j * batch_size:(j + 1) * batch_size]

                        user_adv, cloud_adv, test_loss_temp, Eud_dist= sess.run([x_adv, image_adv, loss, loss_E], feed_dict={images:x, benchmark:ben}) #  final AE
                        if(i==0 and j==0): # 仅在第一个epoch 计算user-end的对抗样本ASR和质量
                            user_acc = sess.run(accuracy, feed_dict={images: user_adv, benchmark: ben})
                            print('Epoch ={} ,User-adv attack acc{:.5} ASR ={:.4} '.format(i, user_acc / int(num_test // batch_size)*100, (1 - user_acc / int(num_test // batch_size)) * 100))
                            cnt = 0
                            # for index in range(batch_size):
                            #     ss_user = []
                            #     ps_user = []
                            #     im_input = cv2.cvtColor(x[index].astype('float32'), cv2.COLOR_BGR2GRAY)  # batch input images
                            #     im_user = cv2.cvtColor(user_adv[index].astype('float32'),cv2.COLOR_BGR2GRAY)  # batch user probe images
                            #     saveplace = os.path.split(args.train_model_output)[0]
                            #     if(cnt<10):
                            #         misc.imsave(saveplace + '/AE/{}.png'.format(cnt), user_adv[index])  # todo # misc.imsave(saveplace + '/Noise/{}.png'.format(cnt), user_adv[index] - x[index])
                            #
                            #     s_t = ssim(im_input, im_user,data_range=1.0)  #  datarange?
                            #
                            #     ss_user.append(s_t)
                            #     p_t = psnr(im_input, im_user,data_range=1.0)#  datarange?
                            #     if (p_t < 500):
                            #         ps_user.append(p_t)
                            #         cnt += 1
                            # s_user = np.mean(ss_user)
                            # s_user_stand = np.std(ss_user)
                            # p_user = np.mean(ps_user)
                            # p_user_stand = np.std(ps_user)
                            # print("User AE  Average SSIM : {:.2}±{:.2}  PSNR : {:.4}±{:.4} ".format(s_user,s_user_stand,p_user,p_user_stand))


                        test_loss += test_loss_temp
                        att_acc += sess.run(accuracy, feed_dict={images: cloud_adv, benchmark: ben})
                        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    print('当前时间 {}  Epoch = {}, 其中iter = {}, training_loss={:.4}, test_loss={:.4},Eudliean_dist={:.4}，对抗攻击后的识别准确率acc = {:.4}'
                          .format(time, i, batch, train_loss, test_loss/int(num_test//batch_size),-Eud_dist + 5, att_acc/int(num_test//batch_size)))
                    for var in tf.trainable_variables():
                        if (var.name.split(':')[0].split('/')[-1] == "Variable"):  # <tf.Variable 'APF/Variable:0' shape=() dtype=float32_ref>
                            # <tf.Variable 'APF/Variable_1:0' shape=() dtype=float32_ref>
                            print(sess.run(var))
                        if (var.name.split(':')[0].split('/')[-1] == "Variable_1"):
                            print(sess.run(var))
                    temp = att_acc / int(num_test // batch_size)
                    if temp < best_acc or (temp==best_acc and train_loss < best_train): # 若攻击效果更好
                        best_train = train_loss
                        best_acc = temp
                        saver_unet.save(sess,args.train_model_output+str(1-best_acc) +'-'+str(i)+'_'+str(j)+'.ckpt')
                        # saver_unet.save(sess,'/data/jiaming/code/InsightFace-tensorflow/model/mm/test_zx/model_apf' + str(best_acc) +'.ckpt')
                        print('-----best_ASR-----  ',(1-best_acc)*100,'%','  train loss:',train_loss)

