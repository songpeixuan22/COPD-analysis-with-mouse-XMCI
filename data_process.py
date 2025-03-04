#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_process.py
~~~~~~~~~~~~~~~~~~~~~
This script uses the dark-field slices of CT image to generate the .csv feature files.

:author: Peixuan Song
:email: spx22@mails.tsinghua.edu.cn
:copyright: (c) 2025 by Peixuan Song.
:license: MIT License, see LICENSE for more details.
"""

__version__ = "0.1.0"

import h5py
import cv2
import glob
import imageio
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

pic_path = 'temp_pic/'
npy_path = 'temp_npy/'
csv_path = 'temp_csv/'

def read_sample(path, axis='x', slice=112):
    '''
    read sample (h5 form)
    ----------
    Parameters:
    path: the path of the h5 file;
    axis: the axis of the slice;
    slice: the slice number of the axis;
    ----------
    Outputs:
    a: the absorption array;
    df: the dark-field array;
    '''
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        a = f[keys[0]]
        df = f[keys[1]]
        if axis == 'z':
            a = a[slice,:,:]
            df = df[slice,:,:]
        elif axis == 'y':
            a = a[:,slice,:]
            df = df[:,slice,:]
        elif axis == 'x':
            a = a[:,:,slice]
            df = df[:,:,slice]
        else:
            raise ValueError('Invalid axis')
        return a, df

def read_sample2(path):
    '''
    read sample (png form)
    ----------
    Parameters:
    path: the path of the png file;
    ----------
    Outputs:
    the array;
    '''
    return np.array(Image.open(path))

def read_sample3(path):
    '''
    read sample (tif form)
    ----------
    Parameters:
    path: the path of the tif file;
    ----------
    Outputs:
    the array;
    '''
    return imageio.imread(path)

def points_cloud(image, threshold, n_clusters=2, random_state=0):
    '''
    To generate the points cloud of the sick area, 
    using manually segmentation and kmeans clusters;
    ----------
    Parameters:
    threshold: to find sick area;
    n_clusters, random_state: to use kmeans;
    ----------
    Outputs:
    picture dictionary
    keys 0,1...: the points of different kmeans labels;
    '''
    dic = {}
    pic_dic = {}
    image_ = np.zeros(image.shape)
    image_[(image > threshold[0]) & (image < threshold[1])] = 1
    x, y = np.where(image_ == 1)
    X = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    
    kmean = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    labels = kmean.labels_

    # Create an array of the same size as the input image to store the points
    point_image = np.zeros_like(image, dtype=int)  # Initialize a zero array with the same shape as the image

    for idx, label in enumerate(labels):
        if label not in dic:
            dic[label] = []
        dic[label].append(X[idx])

        # Mark the corresponding points in the point_image as 1
        point_image[X[idx][0], X[idx][1]] = 1  # Set the corresponding (x, y) point to 1

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for label in dic.keys():
        Xl = np.array(dic[label])
        color = 'yellow' if label == 0 else 'orange'
        ax.scatter(Xl[:, 1], Xl[:, 0], s=0.7, c=color)
        Xl = Xl.astype(np.int8)
        pic_dic[label] = image[np.min(X[:, 0]):np.max(X[:, 0]), np.min(X[:, 1]):np.max(X[:, 1])]

    # Save the output image with points overlaid
    plt.savefig(pic_path+'seg.png')

    return fig, pic_dic, point_image  # Return the point_image along with other outputs


def gaussian_pyramid(image, levels):
    '''
    generate gaussian pyramid
    ----------
    Parameters:
    image: the input image;
    levels: the number of pyramid levels;
    ----------
    Outputs:
    pyramid: the gaussian pyramid;
    '''
    pyramid = [image]
    for _ in range(levels-1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid
def laplacian_pyramid(gp):
    '''
    generate laplacian pyramid
    ----------
    Parameters:
    gp: the gaussian pyramid;
    ----------
    Outputs:
    lp: the laplacian pyramid;
    '''
    lp = []
    for i in range(len(gp) - 1, 0, -1):
        upsampled = cv2.pyrUp(gp[i])  # upsampling
        # resize the upsampled image to the size of the original image
        upsampled = cv2.resize(upsampled, (gp[i-1].shape[1], gp[i-1].shape[0]))
        laplacian = cv2.subtract(gp[i-1], upsampled)  # calculate the Laplacian
        lp.append(laplacian)
    lp.append(gp[0])

    return lp

def histogram(image):
    '''
    calculate the histogram of the image
    ----------
    Parameters:
    image: the input image;
    ----------
    Outputs:
    hist: the histogram;
    mean: the mean of the image;
    std: the std of the image;
    '''
    hist = image[image != 0] # delete the background
    mean = np.mean(image) # mean
    std = np.std(image) # std
    return hist, mean, std

def fft(image):
    '''
    calculate the fft of the image
    ----------
    Parameters:
    image: the input image;
    ----------
    Outputs:
    fshift: the fft of the image;
    as_mean: the mean of the amplitude spectrum;
    as_std: the std of the amplitude spectrum;
    es_mean: the mean of the energy spectrum;
    es_std: the std of the energy spectrum;
    entropy_spectrum: the entropy of the amplitude spectrum;
    spectral_flatness: the spectral flatness;
    '''
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # calculate the amplitude spectrum
    amplitude_spectrum = np.abs(fshift)
    energy_spectrum = amplitude_spectrum ** 2
    # calculate the mean and std of the amplitude spectrum
    as_mean, as_std = np.mean(amplitude_spectrum), np.std(amplitude_spectrum)
    es_mean, es_std = np.mean(energy_spectrum), np.std(energy_spectrum)
    # calculate the entropy of the amplitude spectrum
    normalized_amplitude = amplitude_spectrum / np.sum(amplitude_spectrum)
    entropy_spectrum = -np.sum(normalized_amplitude * np.log2(normalized_amplitude + 1e-10))
    # calculate the spectral flatness
    spectral_flatness = np.exp(np.mean(np.log(amplitude_spectrum + 1e-10))) / (np.mean(amplitude_spectrum) + 1e-10)

    return fshift, as_mean, as_std, es_mean, es_std, entropy_spectrum, spectral_flatness

def glcm(image, d, theta):
    '''
    calculate the glcm of the image
    ----------
    Parameters:
    image: the input image;
    d: the distance;
    theta: the angle;
    ----------
    Outputs:
    contrast: the contrast of the glcm;
    dissimilarity: the dissimilarity of the glcm;
    homogeneity: the homogeneity of the glcm;
    energy: the energy of the glcm;
    correlation: the correlation of the glcm;
    asm: the asm of the glcm;
    glcm_mean: the mean of the glcm;
    glcm_std: the std of the glcm;
    glcm_vari: the variance of the glcm;
    glcm_entropy: the entropy of the glcm;
    '''
    minn = np.min(image)
    maxx = np.max(image)
    image = (image - minn) / (maxx - minn) * 255
    image = image.astype(np.uint8)

    glcm = graycomatrix(image, distances=d, angles=theta,
    levels=256,
    symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    asm = graycoprops(glcm, 'ASM')
    glcm_mean = graycoprops(glcm, 'mean')
    glcm_std = graycoprops(glcm, 'std')
    glcm_vari = graycoprops(glcm, 'variance')
    glcm_entropy = graycoprops(glcm, 'entropy')

    return contrast, dissimilarity, homogeneity, energy, correlation, asm, glcm_mean, glcm_std, glcm_vari, glcm_entropy

def contours(image, threshold=2.55):
    '''
    calculate the contours of the image
    ----------
    Parameters:
    image: the input image;
    threshold: the threshold of the image;
    ----------
    Outputs:
    area: the area of the contours;
    perimeter: the perimeter of the contours;
    hu_moments: the hu moments of the contours;
    diameter: the diameter of the contours;
    '''
    mask = np.where((image>0)&(image<threshold), 255, image).astype(np.uint8)
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area, perimeter, hu_moments, diameter = [], [], [], []

    for i, contour in enumerate(contours):
        area.append(cv2.contourArea(contour))
        perimeter.append(cv2.arcLength(contour, True))
        moments = cv2.moments(contour)
        hu_moments.append(cv2.HuMoments(moments).flatten())
        _,radius = cv2.minEnclosingCircle(contour)
        diameter.append(radius * 2)
    # calculate the mean of the area, perimeter, hu_moments, and diameter
    area = np.mean(area)
    perimeter = np.mean(perimeter)
    hu_moments = np.mean(hu_moments, axis=0)
    diameter = np.mean(diameter)
    return area, perimeter, hu_moments, diameter 


def load_sample(dfs,
              d=np.array([1,2]),
              theta=np.array([0, np.pi/4, np.pi/2, 3*np.pi/4]),
              label=0,
              levels=3):
    '''
    generate csv files
    contains feature vectors
    ----------
    Parameters:
    dfs: the input image;
    d: the distance of the glcm;
    theta: the angle of the glcm;
    label: the label of the image;
    levels: the number of pyramid levels;
    ----------
    Outputs:
    features: the feature vectors;
    '''
    dframe = feature = {'label': label}
    dframe = pd.DataFrame(dframe, index=[0])
    features = []
    dfs = dfs * 255

    gp = gaussian_pyramid(dfs, levels)
    lp = laplacian_pyramid(gp)
    for ii, df in enumerate(lp):
        maxx = np.max(df)
        minn = np.min(df)
        df = (df - minn)/(maxx - minn) * 255

        _, his_mean, his_std = histogram(df)
        _, as_mean, as_std, es_mean, es_std, entropy_spectrum, spectral_flatness = fft(df)
        contrast, dissimilarity, homogeneity, energy, correlation, asm, glcm_mean, glcm_std, glcm_vari, glcm_entropy = glcm(df, d, theta)
        area, perimeter, hu_moments, diameter = contours(df)
        # 生成特征向量
        feature = {
            f'hist_mean_lp{ii}': his_mean,
            f'hist_std_lp{ii}': his_std,
            f'fft_as_mean_lp{ii}': as_mean,
            f'fft_as_std_lp{ii}': as_std,
            f'fft_es_mean_lp{ii}': es_mean,
            f'fft_es_std_lp{ii}': es_std,
            f'fft_entropy_spectrum_lp{ii}': entropy_spectrum,
            f'fft_spectral_flatness_lp{ii}': spectral_flatness,
            f'contour_area_lp{ii}': area,
            f'contour_perimeter_lp{ii}': perimeter,
            f'contour_diameter_lp{ii}': diameter,
            f'contour_hu_moments1_lp{ii}': hu_moments[0],
            f'contour_hu_moments2_lp{ii}': hu_moments[1],
            f'contour_hu_moments3_lp{ii}': hu_moments[2],
            f'contour_hu_moments4_lp{ii}': hu_moments[3],
            f'contour_hu_moments5_lp{ii}': hu_moments[4],
            f'contour_hu_moments6_lp{ii}': hu_moments[5],
            f'contour_hu_moments7_lp{ii}': hu_moments[6],
        }
        for i in range(4):
            feature[f'glcm_contrast_d1_t{i}_lp{ii}'] = contrast[0,i]
            feature[f'glcm_dissimilarity_d1_t{i}_lp{ii}'] = dissimilarity[0,i]
            feature[f'glcm_homogeneity_d1_t{i}_lp{ii}'] = homogeneity[0,i]
            feature[f'glcm_energy_d1_t{i}_lp{ii}'] = energy[0,i]
            feature[f'glcm_correlation_d1_t{i}_lp{ii}'] = correlation[0,i]
            feature[f'glcm_asm_d1_t{i}_lp{ii}'] = asm[0,i]
            feature[f'glcm_mean_d1_t{i}_lp{ii}'] = glcm_mean[0,i]
            feature[f'glcm_std_d1_t{i}_lp{ii}'] = glcm_std[0,i]
            feature[f'glcm_variance_d1_t{i}_lp{ii}'] = glcm_vari[0,i]
            feature[f'glcm_entropy_d1_t{i}_lp{ii}'] = glcm_entropy[0,i]

            feature[f'glcm_contrast_d2_t{i}_lp{ii}'] = contrast[1,i]
            feature[f'glcm_dissimilarity_d2_t{i}_lp{ii}'] = dissimilarity[1,i]
            feature[f'glcm_homogeneity_d2_t{i}_lp{ii}'] = homogeneity[1,i]
            feature[f'glcm_energy_d2_t{i}_lp{ii}'] = energy[1,i]
            feature[f'glcm_correlation_d2_t{i}_lp{ii}'] = correlation[1,i]
            feature[f'glcm_asm_d2_t{i}_lp{ii}'] = asm[1,i]
            feature[f'glcm_mean_d2_t{i}_lp{ii}'] = glcm_mean[1,i]
            feature[f'glcm_std_d2_t{i}_lp{ii}'] = glcm_std[1,i]
            feature[f'glcm_variance_d2_t{i}_lp{ii}'] = glcm_vari[1,i]
            feature[f'glcm_entropy_d2_t{i}_lp{ii}'] = glcm_entropy[1,i]
        feature = pd.DataFrame(feature, index=[0])
        features.append(feature)

    features = pd.concat(features, axis=1)
    features = pd.concat([dframe, features], axis=1)
    return features

def merge_csv_files(csv_files, output_path):
    """
    Merge multiple CSV files into a single CSV file

    :param csv_files: List of input CSV file paths to be merged
    :param output_path: Path where the merged CSV file will be saved
    """
    # read all CSV files into a list of DataFrames
    dfs = [pd.read_csv(file) for file in csv_files]
    # merge all DataFrames in the list
    merged_df = pd.concat(dfs, axis=0)
    # save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)

    print(f"CSV file has successfully saved to {output_path}")

if __name__ == '__main__':  
    postions = ['left_normal', 'right_normal', 'left_abnormal', 'right_abnormal']
    num_left = -1
    num_right = -1
    
    for pos in postions:
        # define the folder path
        folder_path = f'data_processed/{pos}/*.tif'
        # find all files in the folder
        files = sorted(glob.glob(folder_path))
        # create an empty DataFrame
        concatenated_df = pd.DataFrame()

        # 遍历所有文件
        for i, file_path in enumerate(files):
            print(f'pos = {pos}, i = {i}')
            # if (i == 6 and pos == 'left') or (i == 15 and pos == 'right'):
            #     continue

            # read the sample
            img = read_sample3(file_path)
            # get the points cloud
            # _, dic, points = points_cloud(img, [1e-5, 100])
            # np.save(npy_path+'points.npy',points)
            # np.save(npy_path+'img.npy', img)
            # break

            # load the data
            if pos == 'left_normal' or pos == 'right_normal':
                f = load_sample(img, label=0)
            # elif pos == 'left_abnormal':
            #     num_left += 1
            #     if(num_left < 150):
            #         f = load_sample(img, label=0)
            #     elif(num_left%5 == 1 or num_left%5 == 3):
            #         f = load_sample(img, label=1)
            #     else:
            #         continue
            # elif pos == 'right_abnormal':
            #     num_right += 1
            #     if(num_right < 150):
            #         f = load_sample(img, label=0)
            #     elif(num_right%5 == 1 or num_right%5 == 3):
            #         f = load_sample(img, label=1)
            #     else:
            #         continue
            elif pos == 'left_abnormal':
                num_left += 1
                if(num_left < 600):
                    f = load_sample(img, label=int(num_left/150)+1)
                else:
                    continue
                    f = load_sample(img, label=int((num_left-600)/20)+5)
            elif pos == 'right_abnormal':
                num_right += 1
                if(num_right < 600):
                    f = load_sample(img, label=int(num_right/150)+1)
                else:
                    continue
                    f = load_sample(img, label=int((num_right-600)/20)+5)
            # concatenate the DataFrame
            concatenated_df = pd.concat([concatenated_df, f], ignore_index=True)

        # save the DataFrame to a CSV file
        concatenated_df.to_csv(csv_path+f'data_{pos}.csv', index=False)

    merge_csv_files([csv_path + 'data_left_normal.csv', csv_path + 'data_left_abnormal.csv'], csv_path + 'total_data_left.csv')
    merge_csv_files([csv_path + 'data_right_normal.csv', csv_path + 'data_right_abnormal.csv'], csv_path + 'total_data_right.csv')

    

