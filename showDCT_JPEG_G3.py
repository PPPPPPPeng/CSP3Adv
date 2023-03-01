import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import numpy as np
from matplotlib import style
print(matplotlib.matplotlib_fname())
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
''' define  : zigzag 扫描
    input   : 二维矩阵, shape: (row, col)
    output  : 列表, shape: (row*col,)
    variable: k 列表序号, i 行序号, j 列序号, row 行数, col 列数
    method  : 假设 (0, 0) 在左上角, (row-1, col-1) 在右下角的情况. 考虑非边界的情况, 只有右上/左下两个方向.
              以从 (0, 0) 先向右(下)为例, 则会有 i+j 为偶数时右上(左下)前进, 为奇数时左下(右上)的情况前进.
              如果遇到边界, 某个方向收到限制, 移动允许的直线方向'''

img = cv2.imread("/home/yl/PycharmProjects/sumail/APF/论文选图/original.png",0)
j=cv2.imread("/home/yl/PycharmProjects/sumail/APF/论文选图/80.jpeg",0)
g3 = cv2.imread("/home/yl/PycharmProjects/sumail/APF/论文选图/g3.png",0)
# noise = cv2.imread("/home/yl/PycharmProjects/sumail/APF/论文选图/0.png",0)
# 数据类型转换 转换为浮点型
img = img.astype(np.float)
g3 = g3.astype(np.float)
j=j.astype(np.float)
# noise=noise.astype(np.float)

def zigzag(data):
    row = data.shape[0]
    col = data.shape[1]
    num = row * col
    list = np.zeros(num, )
    k = 0
    i = 0
    j = 0

    while i < row and j < col and k < num:
        list[k] = data.item(i, j)
        k = k + 1
        # i + j 为偶数, 右上移动. 下面情况是可以合并的, 但是为了方便理解, 分开写
        if (i + j) % 2 == 0:
            # 右边界超出, 则向下
            if (i - 1) in range(row) and (j + 1) not in range(col):
                i = i + 1
            # 上边界超出, 则向右
            elif (i - 1) not in range(row) and (j + 1) in range(col):
                j = j + 1
            # 上右边界都超出, 即处于右上顶点的位置, 则向下
            elif (i - 1) not in range(row) and (j + 1) not in range(col):
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        # i + j 为奇数, 左下移动
        elif (i + j) % 2 == 1:
            # 左边界超出, 则向下
            if (i + 1) in range(row) and (j - 1) not in range(col):
                i = i + 1
            # 下边界超出, 则向右
            elif (i + 1) not in range(row) and (j - 1) in range(col):
                j = j + 1
            # 左下边界都超出, 即处于左下顶点的位置, 则向右
            elif (i + 1) not in range(row) and (j - 1) not in range(col):
                j = j + 1
            else:
                i = i + 1
                j = j - 1

    return list

def block_img_dct(img_f32):
    height,width = img_f32.shape[:2]
    block_y = height // 8
    block_x = width // 8
    height_ = block_y * 8
    width_ = block_x * 8
    img_f32_cut = img_f32[:height_, :width_]
    img_dct = np.zeros((8, 8), dtype=np.float32)
    # for h in range(block_y): # 0~14
    #     for w in range(block_x):# 0~14
            # 对图像块进行dct变换
    i = 8  # 4
    j = 6  # 4
    img_block = img_f32_cut[8*i: 8*(i+1), 8*j: 8*(j+1)] # specific block
            # img_block = img_f32_cut[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)]  #
    img_dct = cv2.dct(img_block)

    return img_dct

def abs_zigzag(result_list):
    result_abs_list = []
    for i in result_list:
        result_abs_list.append(abs(i))
    return result_abs_list

b_img = block_img_dct(img)
b_j = block_img_dct(j)
b_g3 = block_img_dct(g3)
# b_noise = block_img_dct(noise)

plt.figure(3,figsize = (12,8))
plt.subplot(131)
plt.title('Image DCT block')
plt.imshow(b_img,'Reds')
plt.colorbar()
plt.subplot(132)
plt.title('JPEG DCT block')
plt.imshow(b_j,'Reds')
# plt.colorbar()
plt.subplot(133)
plt.title('Guassian DCT block')
plt.imshow(b_g3,cmap='Reds')
# plt.colorbar()
# plt.imshow(b_noise,'gray')
plt.show()


r_img = zigzag(b_img)  # np.array 格式的输出
a_img = abs_zigzag(r_img)
r_j = zigzag(b_j)  # np.array 格式的输出
a_j = abs_zigzag(r_j)
r_g3 = zigzag(b_g3)  # np.array 格式的输出
a_g3 = abs_zigzag(r_g3)
# r_noise = zigzag(b_noise)  # np.array 格式的输出
# a_noise = abs_zigzag(r_noise)

style.use('ggplot')
plt.figure(3,figsize = (15,10))
plt.subplot(131)
plt.bar(range(len(a_img)),a_img)
plt.title('Img frequency histogram')
plt.subplot(132)
plt.bar(range(len(a_j)),a_j)
plt.title('JPEG frequency histogram')
plt.subplot(133)
plt.bar(range(len(a_g3)),a_g3)
# plt.bar(range(len(a_noise)),a_noise)
plt.title('Guassian frequency histogram')
plt.show()


# img = cv2.imread("/home/yl/PycharmProjects/sumail/APF/论文选图/original.png",0)
# j=cv2.imread("/home/yl/PycharmProjects/sumail/APF/论文选图/JPEG66after.jpg",0)
# g3 = cv2.imread("/home/yl/PycharmProjects/sumail/APF/论文选图/g3.png",0)
# # 数据类型转换 转换为浮点型
# # print('0\n',img)
# img1 = img.astype(np.float)
# g3 = g3.astype(np.float)
# j=j.astype(np.float)
# g3_dct = cv2.dct(g3)
# g3_dct_log = np.log(abs(g3_dct))
# j_dct = cv2.dct(j)
# j_dct_log = np.log(abs(j_dct))
#
# # 进行离散余弦变换
# img_dct = cv2.dct(img1)
# # print('1\n',img_dct)
# # 进行log处理
# img_dct_log = np.log(abs(img_dct))
# # print('2\n',img_dct_log)
# # 进行离散余弦反变换
#
# y_dct = img_dct.flatten()
# print('flatten jpeg:',y_dct)
# y_jpeg = j_dct.flatten()
# print('flatten jpeg:',y_jpeg)
# y_g3 = g3_dct.flatten()
# print('flatten g3:',y_g3)
#
# plt.subplot(141)
# plt.imshow(img,'gray')
# plt.title('original image'),plt.xticks([]),plt.yticks([])
# plt.subplot(142)
# plt.imshow(img_dct, 'gray')
# plt.title('DCT'),plt.xticks([]),plt.yticks([])
# plt.subplot(143)
# plt.imshow(j_dct,'gray')
# plt.title('JPEG'),plt.xticks([]),plt.yticks([])
# print('JPEG:\n',j_dct)
# plt.subplot(144)
# plt.imshow(g3_dct,'gray')
# plt.title('Guassian'),plt.xticks([]),plt.yticks([])
# print('Guassian:\n',g3_dct)
# print(j_dct.shape)
# plt.show()

