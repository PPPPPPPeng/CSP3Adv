# 读取所有tfrecord文件得到dataset
# create training data for APF, which includes 100,000 face images of 2,000 subjects (you can generate more training data if you have bigger memory)
# 由于内存限制，只选择了1500个对象的75000张图片
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from utils.util import *



# 最终的接口是 create_data（）


# 解析dataset的函数, 直接把bytes转换回image, 对应方法1
# def parse_record(raw_record):
#     # 按什么格式写入的, 就要以同样的格式输出
#     keys_to_features = {
#         'image': tf.FixedLenFeature((), tf.string),
#         'label': tf.FixedLenFeature((), tf.string),
#     }
#     # 按照keys_to_features解析二进制的
#
#
#     parsed = tf.parse_single_example(raw_record, keys_to_features)
#
#     image = tf.image.decode_image(tf.reshape(parsed['image'], shape=[112,112,3]), 1)
#     image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
#     image.set_shape([None, None, 1])
#     label = tf.image.decode_image(tf.reshape(parsed['label'], shape=[]), 1)
#     label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
#     label.set_shape([None, None, 1])
#
#     return image, label

#直接把bytes类型的ndarray解析回来, 用decode_raw(),对应方法2
def parse_record(raw_record):
    keys_to_features = {
        'img': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64),
        'shape':tf.FixedLenFeature([3],tf.int64)
    }
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    #
    #     img = tf.reshape(img, shape=(112, 112, 3))

    image = tf.decode_raw(parsed['img'],tf.uint8)
    # image = tf.to_float(image)
    image = tf.reshape(image, [112, 112, 3])
    label = tf.cast(parsed['label'], tf.int64)
    # label = tf.decode_raw(parsed['label'], tf.uint8)
    # label = tf.to_int32(label)
    # label = tf.reshape(label, [256, 256, 1])

    return image, label


# 根据图片数量还原图片

def generate_image(sess, image_num, read_path):

    dataset = tf.data.TFRecordDataset(read_path)
    # 对dataset中的每条数据, 应用parse_record函数, 得到解析后的新的dataset
    dataset = dataset.map(parse_record)
    dataset = dataset.batch(image_num)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # sess.run(iterator.initializer)
    images, labels = sess.run(next_element)
    return images,labels


# 图像类别是0、1、2、3。。。。。80000多
# 这个是最终函数，传某一类别值（0～84000+），返回这一类别的所有图像
def create_data(sess, class_num):
    images,labels = generate_image(sess, (class_num+1)*200)
    # images = np.zeros([class_num*batch_num,112,112,3])
    indicates = np.where(labels == class_num)
    imgs = images[indicates]
    return imgs,imgs.shape[0]


def pickle_save(sess, col, path, read_path): # col =2000

    images = []
    cnt = col*2 #预选的对象范围，因为不是每个人的图像数量都超过50张，所以控制两倍col的人数区间进行选择
    imgs,labels = generate_image(sess,cnt*200, read_path) # 控制预读取的图片总数
    # images:对象总数
    temp = 0
    print("start  save")
    # 第一次500个对象选择
    for i in range(cnt):
        indicates = np.where(labels == i)
        img = imgs[indicates]
        if img.shape[0] >= 50:# 每个主体的图像上限为默认=50 : 2,000*50=100,000
            image = img[:50]
            images.append(image)
            temp += 1
            print('[',i,']',"进度：",temp,'/',col,'    ',img.shape[0],'---',len(image))
        if(temp>=col):
            step1 = i
            break
    print('第一次选择完成后的当前对象下标为：',step1)
    #第二次500对象选择
    for i in range(step1,step1+cnt):
        indicates = np.where(labels == i) # 表示当前对象的图像下标范围
        img = imgs[indicates]
        if img.shape[0] >= 50:# 每个主体的图像上限为默认=50 : 2,000*50=100,000
            image = img[:50]
            images.append(image)
            temp += 1
            print('[',i,']',"进度：",temp,'/',col+500,'    ',img.shape[0],'---',len(image))
        if(temp>=500+col):
            step2 = i
            break
    print('第二次选择完成后的当前对象下标为：', step2)
    # 第三次500对象选择
    for i in range(step2, step2 + cnt):
        indicates = np.where(labels == i)  # 表示当前对象的图像下标范围
        img = imgs[indicates]
        if img.shape[0] >= 50:  # 每个主体的图像上限为默认=50 : 2,000*50=100,000
            image = img[:50]
            images.append(image)
            temp += 1
            print('[', i, ']', "进度：", temp, '/', col + 1000, '    ', img.shape[0], '---', len(image))
        if (temp >= 1000 + col):
            step3 = i
            break
    print('第三次选择完成后的当前对象下标为：', step3)

    print('总计选择对象个数：',len(images))
    print(labels)

    with open(path,'wb') as f:
        pickle.dump(images,f)
    print('finish ')
    return images



def pickle_load(path):

    with open(path,'rb') as f:
        images = pickle.load(f)
    return images

# 前10类的图像的shape如下所示：
#(110, 112, 112, 3)
# (19, 112, 112, 3)
# (83, 112, 112, 3)
# (15, 112, 112, 3)
# (75, 112, 112, 3)
# (82, 112, 112, 3)
# (103, 112, 112, 3)
# (88, 112, 112, 3)
# (96, 112, 112, 3)
# (35, 112, 112, 3)
# 可以看出来每一类图片数量不一样且差别很大
# 我本来想固定batch_size，让每一类的图像数目相同，但是这样就会浪费很多图片（因为要取数量的最小值）
# 所以现在这个create_data只能返回某一类的图

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--subjects', type=int, help='number of subjects', default=2000) #  主体数量
    parser.add_argument('--read_path', type=str, help='TFRecord file', default='') # 读取路径
    parser.add_argument('--save_path', type=str, help='path to save', default='') # 保存路径

    return parser.parse_args()

if __name__ == '__main__':
    with tf.Session() as sess:
        args = get_args()
        path = args.save_path
        pickle_save(sess, args.subjects, path, args.read_path)

