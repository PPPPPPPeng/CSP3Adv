import numpy as np
import os
import pickle
from scipy import misc
import io
import imageio
from PIL import Image
import cv2
from scipy import misc
def create_lfw_npy_train(path='/media/yl/东方芝士/DataSets/MS-Celeb-1M-ArcFace(InsightFace)/faces_emore/lfw.bin',image_size=112):
    # 3000 pairs include same and insame pairs
    print('正在读取测试集文件 : reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    images = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    images_f = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    for bin in bins:
        img = imageio.imread(io.BytesIO(bin))
        img = np.array(Image.fromarray(img).resize([image_size, image_size]))

        # img = img[s:s+image_size, s:s+image_size, :]
        img = img / 127.5 - 1.0
        images[cnt] = img
        cnt += 1
    print('测试集图片准备完成!')
    print('bin中的图片总数 ' + str(num)+',选择其中前6000张作为测试数据')
    print("picture:",images[0])

    images = images
    images_copy = np.zeros((6000, image_size, image_size, 3))
    for i in range(10):
        images_copy[600 * i:600 * (i + 1)] = images[600 * i * 2:600 * (i * 2 + 1)]
    return images_copy

def create_lfw_npy_samepair(path='/media/yl/东方芝士/DataSets/MS-Celeb-1M-ArcFace(InsightFace)/faces_emore/lfw.bin',image_size=112):
    print('正在读取测试集文件 : reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    images = np.zeros((6000, image_size, image_size, 3))
    for i, issame in enumerate(issame_list):
        # img0 = Image.open(io.BytesIO(bins[2 * i]))
        # img1 = Image.open(io.BytesIO(bins[2 * i + 1]))
        # img0 = img0.resize((image_size, image_size))
        # img1 = img1.resize((image_size, image_size))
        # print(img0)
        img0 = misc.imread(io.BytesIO(bins[2 * i]))
        img1 = misc.imread(io.BytesIO(bins[2 * i + 1]))
        img0 = misc.imresize(img0, [image_size, image_size])
        img1 = misc.imresize(img1, [image_size, image_size])

        img0 = img0 / 127.5 - 1.0
        img1 = img1 / 127.5 - 1.0
        if (issame):
            # print(issame)
            images[cnt:cnt + 1] = img0
            images[cnt + 1:cnt + 2] = img1
            # if cnt < 6000:
                # img0.save('/home/yl/PycharmProjects/sumail/APF/data/lfw_same_jpeg/{}.jpeg'.format(cnt),quality = 100, subsampling = 0, format = 'JPEG')
                # img1.save('/home/yl/PycharmProjects/sumail/APF/data/lfw_same_jpeg/{}.jpeg'.format(cnt+1), quality=100, subsampling=0, format='JPEG')
                # misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/lfw_same_bmp/{}.bmp'.format(cnt), img0)
                # misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/lfw_same_bmp/{}.bmp'.format(cnt+1), img1)
            cnt += 2

    # for bin in bins:
    #     img = imageio.imread(io.BytesIO(bin))
    #     # img: array without name
    #     img = np.array(Image.fromarray(img).resize([image_size, image_size]))
    #     img = img / 127.5 - 1.0  #[-1,1]
    #     images[cnt] = img
    #     # if cnt < 6000:
    #         # misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/lfw/{}.png'.format(cnt), img)
    #     cnt += 1
    print('测试集图片准备完成!')
    print('bin中的图片总数 ' + str(num)+',选择其中前6000张作为测试数据')
    # print(images[0])
    return images

def create_calfw_npy(path='/media/yl/东方芝士/DataSets/MS-Celeb-1M-ArcFace(InsightFace)/faces_emore/calfw.bin',image_size=112):
    print('正在读取测试集文件 : reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    images = np.zeros((6000, image_size, image_size, 3))
    for i, issame in enumerate(issame_list):

        img0 = misc.imread(io.BytesIO(bins[2 * i]))
        img1 = misc.imread(io.BytesIO(bins[2 * i + 1]))
        img0 = misc.imresize(img0, [image_size, image_size])
        img1 = misc.imresize(img1, [image_size, image_size])
        # img = img[s:s+image_size, s:s+image_size, :]

        img0 = img0 / 127.5 - 1.0
        img1 = img1 / 127.5 - 1.0
        if (issame):
            # print(issame)
            images[cnt:cnt + 1] = img0
            images[cnt + 1:cnt + 2] = img1
            if cnt < 6000:
                misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/calfw_same/{}.png'.format(cnt), img0)
            #     misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/calfw_same/{}.png'.format(cnt + 1), img1)
            cnt += 2
    return images

def create_cplfw_npy(path='/media/yl/东方芝士/DataSets/MS-Celeb-1M-ArcFace(InsightFace)/faces_emore/cplfw.bin',image_size=112):
    print('正在读取测试集文件 : reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    images = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    images_f = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    images = np.zeros((6000, image_size, image_size, 3))
    for i, issame in enumerate(issame_list):
        img0 = misc.imread(io.BytesIO(bins[2 * i]))
        img1 = misc.imread(io.BytesIO(bins[2 * i + 1]))
        img0 = misc.imresize(img0, [image_size, image_size])
        img1 = misc.imresize(img1, [image_size, image_size])

        img0 = img0 / 127.5 - 1.0
        img1 = img1 / 127.5 - 1.0
        if (issame):
            # print(issame)
            images[cnt:cnt + 1] = img0
            images[cnt + 1:cnt + 2] = img1
            # if cnt < 6000:
            #     misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/cplfw_same/{}.png'.format(cnt), img0)
            #     misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/cplfw_same/{}.png'.format(cnt + 1), img1)
            cnt += 2
    return images

def create_cfp_npy(path='/media/yl/东方芝士/DataSets/MS-Celeb-1M-ArcFace(InsightFace)/faces_emore/cfp_fp.bin',image_size=112):
    print('正在读取测试集文件 : reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    total = 0
    imgs = np.zeros((7000,image_size,image_size,3))
    for i,issame in enumerate(issame_list):
        total += 2
        img0 = misc.imread(io.BytesIO(bins[2*i]))
        img1 = misc.imread(io.BytesIO(bins[2*i+1]))
        img0 = misc.imresize(img0, [image_size, image_size])
        img1 = misc.imresize(img1, [image_size, image_size])
        # img = img[s:s+image_size, s:s+image_size, :]

        img0 = img0 / 127.5 - 1.0
        img1 = img1 / 127.5 - 1.0
        if (issame):
            # print(issame)
            imgs[cnt:cnt+1] = img0
            imgs[cnt+1:cnt+2] = img1
            # if cnt < 7000:
            #     misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/cfp_fp/{}.png'.format(cnt), img0)
            #     misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/cfp_fp/{}.png'.format(cnt+1), img1)
            cnt += 2
    print('测试集图片准备完成!')
    print('bin中的图片总数 ' + str(total) + ',选择其中前7000张作为测试数据')
    return imgs

def create_agedb_npy(path='/media/yl/东方芝士/DataSets/MS-Celeb-1M-ArcFace(InsightFace)/faces_emore/agedb_30.bin',image_size=112):
    print('正在读取测试集文件 : reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    total = 0
    imgs = np.zeros((6000,image_size,image_size,3))
    for i,issame in enumerate(issame_list):
        total += 2
        img0 = misc.imread(io.BytesIO(bins[2*i]))
        img1 = misc.imread(io.BytesIO(bins[2*i+1]))
        img0 = misc.imresize(img0, [image_size, image_size])
        img1 = misc.imresize(img1, [image_size, image_size])
        # img = img[s:s+image_size, s:s+image_size, :]

        img0 = img0 / 127.5 - 1.0
        img1 = img1 / 127.5 - 1.0
        if (issame):
            # print(issame)
            imgs[cnt:cnt+1] = img0
            imgs[cnt+1:cnt+2] = img1
            # if cnt < 6000:
            #     misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/agedb_30/{}.png'.format(cnt), img0)
            #     misc.imsave('/home/yl/PycharmProjects/sumail/APF/data/agedb_30/{}.png'.format(cnt+1), img1)
            cnt += 2
    print('测试集图片准备完成!')
    print('bin中的图片总数 ' + str(total) + ',选择其中前6000张作为测试数据')
    return imgs
