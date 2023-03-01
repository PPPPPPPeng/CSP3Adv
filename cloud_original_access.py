from backbones.insightface import *
from utils.util import *
from utils.attack import *
from backbones.unet import unet
from backbones.MobileFaceNet import mobilefacenet

parser = argparse.ArgumentParser()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import cv2
import tensorflow as tf
# tf.enable_eager_execution()
import lpips_tf
import time
output_dir = '/home/yl/PycharmProjects/sumail/APF/data/Ori'
output_noise_dir = '/home/yl/PycharmProjects/sumail/APF/data/AE'

parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
parser.add_argument('--insightface_model_path', type=str, default='/media/yl/东方芝士/pretrained/Insightface_ms1m_100_334k/best-m-334000', help='model path')
args = parser.parse_args()

def insight(inputs):
    embeddings,_ = get_embd(inputs,config)
    return embeddings

def attack_cloud(args):
    batch_size = 100  # config_ms1m_100: 100
    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)  # 加载参数
    benchmark = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_benchmark')  # probe
    images = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_image')  # 加载图像

    arc_embds, _ = get_embd(images, config) # to get the insightface embeddings
    arc_ben_embds, _ = get_embd(benchmark, config)# to get the insightface embeddings


    def get_distance(embds1, embds2 = arc_ben_embds):# square Euclidean distance
        embeddings1 = embds1 / tf.norm(embds1, axis=1, keepdims=True)
        embeddings2 = embds2 / tf.norm(embds2, axis=1, keepdims=True)
        diff = tf.subtract(embeddings1, embeddings2)
        distance = tf.reduce_sum(tf.multiply(diff, diff), axis=1)
        return distance
    # define
    # d = get_distance(arc_embds, arc_ben_embds)
    # x_FGSM = FGSM(images, d)  # ok
    # x_IFGSM = IFGSM(images, lambda f_embd: insight(f_embd),lambda f_dis: get_distance(f_dis), 1) #
    # x_MIFGSM = MIFGSM(images, lambda f_embd: insight(f_embd), lambda f_dis: get_distance(f_dis),1) #
    # x_DI2FGSM = DI2FGSM(images, lambda f_embd: insight(f_embd), lambda f_dis: get_distance(f_dis), 1) #
    x_MDI2FGSM = MDI2FGSM(images, lambda f_embd: insight(f_embd), lambda f_dis: get_distance(f_dis),1) #
    # print("attack done")
    adv = x_MDI2FGSM
    adv = tf.clip_by_value(adv, -1.0, 1.0)

    dist = get_distance(arc_embds, arc_ben_embds)
    threshold = 1.12  # LFW
    # threshold = 1.1 # CALFW
    # threshold = 1.13 # CPLFW
    # threshold = 1.16 # agedb30
    # threshold = 1.29 # cfpfp
    distances = threshold - dist
    prediction = tf.sign(distances)  # distances距离小于threshold即二者属于同一人，prediction = 0,二者为同一人, or else prediction = -1
    correct_prediction = tf.count_nonzero(prediction + 1, dtype=tf.float32)
    accuracy = correct_prediction / batch_size  # 计算ARCFACE、INSIGHTFACE下的准确率:二者为同一人的概率

    X = create_lfw_npy_samepair()  # 读取LFW测试集图像，创建6000张测试集图片
    # X = create_calfw_npy()  # 读取CALFW测试集图像，创建6000张测试集图片
    # X = create_cplfw_npy()  # 读取CPLFW测试集图像，创建6000张测试集图片
    # X = create_agedb_npy()  # 读取AGEDB-30测试集图像，创建6000张测试集图片
    # X = create_cfp_npy()     # 读取CFP-FP测试集图像，创建7000张测试集图片

    print('测试集图片数目，图像大小:', X.shape)
    X_0 = X[0::2]  # 切片，从第0张开始每隔两张图片取出一张图片
    X_1 = X[1::2]  # 切片，从第1张开始每隔两张图片取出一张图片
    num_test = len(X_0)
    print('测试集对象数目:', num_test, 'batch_size:', batch_size)
    wo_acc = 0
    ori_acc = 0
    ss = []
    ps = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    variables_arc = tf.contrib.framework.get_variables_to_restore(include=['embd_extractor'])
    saver_arcface = tf.train.Saver(variables_arc)
    # make images directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_noise_dir):
        os.makedirs(output_noise_dir)

    with tf.Session(config=config) as sess:
        saver_arcface.restore(sess, args.insightface_model_path)
        for j in range(num_test // batch_size):  # 分 30 个 batch进行计算
            print("batch : {}/{}".format(j+1,30))
            x = X_0[j * batch_size:(j + 1) * batch_size]  # batch input images
            ben = X_1[j * batch_size:(j + 1) * batch_size]  # batch probe images
            x_adv = sess.run(adv, feed_dict={images: x, benchmark: ben})
            ori_acc += sess.run(accuracy, feed_dict={images: x, benchmark: ben})
            wo_acc += sess.run(accuracy, feed_dict={images: x_adv, benchmark: ben})  # wo:wrong output 计算对抗攻击后的识别准确率
            for i in range(batch_size):
                # X 已经是np.array格式了
                print("Batch: {}/{} - pic: {}/{}".format(j+1,num_test//batch_size,i+1,batch_size))
                # SSIM\PSNR
                # if j==0:
                #     misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/Ori/{}.png'.format(i), x[i].astype('float32'))
                #     misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/AE/{}.png'.format(i), x_adv[i].astype('float32'))
                im1 = cv2.cvtColor(x[i].astype('float32'), cv2.COLOR_BGR2GRAY)  # batch input images
                im2 = cv2.cvtColor(x_adv[i].astype('float32'), cv2.COLOR_BGR2GRAY)  # batch probe images
                s_temp = ssim(im1, im2)
                if(s_temp!= 1):
                    ss.append(s_temp)
                p_temp = psnr(im1, im2)
                if(p_temp<500):
                    ps.append(p_temp)
                print("SSIM: {} PSNR: {} ".format(s_temp, p_temp))


        print("-------------------")
        s = np.mean(ss)
        s_stand = np.std(ss)
        p = np.mean(ps)
        p_stand = np.std(ps)
        print("Average SSIM : {:.2}±{:.2}  PSNR : {:.4}±{:.4} ".format(s, s_stand, p, p_stand))
        print('Model Ori acc={:.5}'.format(ori_acc / (num_test // batch_size)))
        print('Model AE acc={:.5} ASR ={:.4} '.format(wo_acc / (num_test // batch_size),(1 - wo_acc / (num_test // batch_size)) * 100))


if __name__ == "__main__":
    attack_cloud(args)
    # if args.mode=='test':
    #     print("test start.")
    #     test(args)

