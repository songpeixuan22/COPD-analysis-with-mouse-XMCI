from skimage import io
from skimage.feature import graycomatrix, graycoprops
import numpy as np

# 读取图像
image = np.load('test_raw.npy').astype(np.uint8)
breakpoint()

# 计算图像的灰度共生矩阵
distances = [1,2]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

# 从GLCM中提取纹理属性
contrast = graycoprops(glcm, 'contrast')
dissimilarity = graycoprops(glcm, 'dissimilarity')
homogeneity = graycoprops(glcm, 'homogeneity')
ASM = graycoprops(glcm, 'ASM')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')

print("Contrast:", contrast)
print("Dissimilarity:", dissimilarity)
print("Homogeneity:", homogeneity)
print("ASM:", ASM)
print("Energy:", energy)
print("Correlation:", correlation)
