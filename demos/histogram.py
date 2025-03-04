import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 加载图像
image = np.load('test_raw.npy')
# print(np.unique(image))

# 绘制灰度图像的直方图
plt.figure(figsize=(8, 6))
plt.hist(image.ravel(), bins=256, range=(-0.05, 0.05), fc='k', ec='k')
plt.title('Grayscale Histogram')
plt.xlabel('Grayscale value')
plt.ylabel('Pixel count')
plt.savefig('test_histogram.png')