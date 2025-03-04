import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def thereshold(path, threshold):
    image = np.load(path)
    breakpoint()
    image[image > threshold] = 1
    image[image <= threshold] = 0
    return image

if __name__ == '__main__':
    image = thereshold('test_raw.npy', 0)
    plt.imshow(image, cmap='gray')
    plt.savefig('test_threshold.png')