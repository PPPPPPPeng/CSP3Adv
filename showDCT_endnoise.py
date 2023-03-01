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

before = cv2.imread("/home/yl/PycharmProjects/sumail/APF/local_train_model/1006K-(30)no-UAP-Coord-DI2FGSM/lfwAE/0.png",0)
after=cv2.imread("/home/yl/PycharmProjects/sumail/APF/local_train_model/eps8-G3before-noUAP-Coord-(30)DI2FGSM/lfwAE/0.png",0)

# 数据类型转换 转换为浮点型
before = before.astype(np.float)
after = after.astype(np.float)


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
    for h in range(block_y): # 0~14
        for w in range(block_x):# 0~14
            # 对图像块进行dct变换
            i = 3  # 4
            j = 4  # 4
            img_block = img_f32_cut[8*i: 8*(i+1), 8*j: 8*(j+1)] # specific block
            # img_block = img_f32_cut[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)]  # specific block
            img_dct = cv2.dct(img_block)

    return img_dct

def abs_zigzag(result_list):
    result_abs_list = []
    for i in result_list:
        result_abs_list.append(abs(i))
    return result_abs_list

b_before = block_img_dct(before)
b_after = block_img_dct(after)

plt.figure(3,figsize = (12,8))
plt.subplot(121)
plt.title('w/o Guassian DCT block')
plt.imshow(b_before,'gray')
plt.subplot(122)
plt.title('w/ Guassian DCT block')
plt.imshow(b_after,'gray')
plt.show()


r_before = zigzag(b_before)  # np.array 格式的输出
a_before = abs_zigzag(r_before)
r_after = zigzag(b_after)  # np.array 格式的输出
a_after = abs_zigzag(r_after)


style.use('ggplot')
plt.figure(3,figsize = (15,10))
plt.subplot(121)
plt.bar(range(len(a_before)),a_before)
plt.title('w/o Guassian histogram')
plt.subplot(122)
plt.bar(range(len(a_after)),a_after)
plt.title('w/ Guassian histogram')

plt.show()