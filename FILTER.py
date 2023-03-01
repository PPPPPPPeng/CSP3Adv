import numpy as np
import cv2
import matplotlib.pyplot as plt

# (1)做傅里叶变换，并显示频谱图像
Lena = cv2.imread('/home/yl/PycharmProjects/sumail/APF/论文选图/original.png')
img = cv2.cvtColor(Lena, cv2.COLOR_BGR2GRAY)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)
result = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('original'), plt.axis('off')
# plt.subplot(122), plt.imshow(result, cmap='gray')
# plt.title('original'), plt.axis('off')
# plt.show()


# (2)理想高通滤波器（3）理想低通滤波器
def IdealHighPassFiltering(f_shift):
    # 设置滤波半径
    D0 = 80
    # 初始化
    m = f_shift.shape[0]
    n = f_shift.shape[1]
    h1 = np.zeros((m, n))
    x0 = np.floor(m / 2)
    y0 = np.floor(n / 2)
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
            if D >= D0:
                h1[i][j] = 1
    result = np.multiply(f_shift, h1)
    return result


def IdealLowPassFiltering(f_shift):
    # 设置滤波半径
    D0 = 80
    # 初始化
    m = f_shift.shape[0]
    n = f_shift.shape[1]
    h1 = np.zeros((m, n))
    x0 = np.floor(m / 2)
    y0 = np.floor(n / 2)
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
            if D <= D0:
                h1[i][j] = 1
    result = np.multiply(f_shift, h1)
    return result


f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)

# 理想高通滤波
# IHPF = IdealHighPassFiltering(f_shift)
# new_f1 = np.fft.ifftshift(IHPF)
# new_image1 = np.uint8(np.abs(np.fft.ifft2(new_f1)))
# plt.subplot(1, 2, 1)
# plt.imshow(new_image1, 'gray')
# 理想低通滤波
GLPF = IdealLowPassFiltering(f_shift)
new_f2 = np.fft.ifftshift(GLPF)
new_image2 = np.uint8(np.abs(np.fft.ifft2(new_f2)))
plt.subplot(1, 2, 2)
plt.imshow(new_image2, 'gray')
plt.show()