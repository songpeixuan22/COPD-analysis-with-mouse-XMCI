import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def gaussian_pyramid(image, levels):
    f = 2**levels
    x, y = image.shape
    x, y = int(x/f)*f, int(y/f)*f
    image = cv2.resize(image, (x, y), interpolation=cv2.INTER_AREA)

    pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def laplacian_pyramid(gaussian_pyramid):
    pyramid = []
    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        # 上采样
        expanded_image = cv2.pyrUp(gaussian_pyramid[i])
        # 计算差值
        diff = cv2.subtract(gaussian_pyramid[i - 1], expanded_image)
        pyramid.append(diff)
    pyramid.append(gaussian_pyramid[0])  # 最后一层是原始图像
    return pyramid

# 加载图像
# image = np.load('test_raw.npy')
image = np.array(Image.open('a.png'))
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

# 构建高斯金字塔
levels = 4 
gaussian_pyramid = gaussian_pyramid(image, levels)
# 构建拉普拉斯金字塔
laplacian_pyramid = laplacian_pyramid(gaussian_pyramid)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i in range(4):
    row = i // 2
    col = i % 2
    axs[row, col].imshow(laplacian_pyramid[i], cmap='gray')
    axs[row, col].axis('off')  # 关闭坐标轴

plt.savefig('test_pyramid.png')
