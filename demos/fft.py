import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy import ndimage
from scipy.fftpack import fft2, fftshift

# 读取图像数据
image = np.load('test_raw.npy')

# 进行二维傅里叶变换
f = fft2(image)
fshift = fftshift(f)

# 计算幅度谱
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# 计算纹理特征
contrast = ndimage.generic_filter(image, np.std, size=3)
energy = ndimage.generic_filter(image, lambda x: np.sum(x**2), size=3)

# 打印纹理特征
print("Contrast:")
print(contrast)
print("\nEnergy:")
print(energy)

# 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.savefig('test_fft.png')

plt.show()
