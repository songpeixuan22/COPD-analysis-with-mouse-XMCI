#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seg_unet.py
~~~~~~~~~~~~~~~~~~~~~
This script uses the pre-trained U-Net to segment the lung picture.

:author: Peixuan Song
:email: spx22@mails.tsinghua.edu.cn
:copyright: (c) 2025 by Peixuan Song.
:license: MIT License, see LICENSE for more details.
"""

__version__ = "0.1.0"

import os
import cv2
import warnings
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

import utils.network as nt
from utils.dataloader import CustomDataset2

warnings.filterwarnings('ignore')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# preprocess
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# load data
data_dir = 'dataset'
dataset = CustomDataset2(data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# load model
model = torch.load('seg1.pth')

def rename(tuple, dst):
    '''
    Rename the file
    ----------------
    tuple: tuple
        The tuple contains the source path and the label path.
        dst: str
        The destination folder.
    ----------------
    return: tuple
        The tuple contains the new source path and the new label path.
    '''
    source_path = tuple[0]
    filename = os.path.basename(source_path)
    fs = filename.split('_')

    subf = fs[0] + '_' + fs[1]
    file = fs[2] + '_' + fs[3]

    new_dir = os.path.join(dst, subf)
    l, r = os.path.join(new_dir, 'l'), os.path.join(new_dir, 'r')
    os.makedirs(l, exist_ok=True)
    os.makedirs(r, exist_ok=True)

    return os.path.join(l, file), os.path.join(r, file)

def rename2(tuple, dst):
    '''
    Rename the file
    ----------------
    tuple: tuple
        The tuple contains the source path and the label path.
        dst: str
        The destination folder.
    ----------------
    return: tuple
        The tuple contains the new source path and the new label path.'''
    source_path = tuple[0]
    filename = os.path.basename(source_path)
    fs = filename.split('_')

    subf = fs[0] + '_' + fs[1]
    file = fs[2] + '_' + fs[3]

    new_dir = os.path.join(dst, subf)
    final = os.path.join(new_dir, 'df')
    os.makedirs(final, exist_ok=True)

    return os.path.join(final, file)

def mask_repair(mask):
    '''
    To find the true lung without noise.
    ----------------
    mask: np.array
        The mask of the lung.
    ----------------
    return: np.array
        The repaired mask.
    '''
    mask[mask == 1] = 255
    mask = mask.astype(np.uint8)
    
    # find all the contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # find the biggest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    h, w = mask.shape
    result_mask = np.zeros((h, w), dtype=np.uint8)
    if max_contour is not None:
        cv2.drawContours(result_mask, [max_contour], 0, 255, cv2.FILLED)
    
    result_mask[result_mask==255] = 1
    return result_mask 

for index, (image, path) in tqdm(enumerate(train_loader), total=len(train_loader)):
    df_src = rename2(path, 'data_tif')
    l_save, r_save = rename(path, 'data_groundtruth_')

    # forward
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        output = F.interpolate(output, size=(1000,1000), mode='bilinear')


    # binary
    output_binary = (output > 0.5).float()

    # numpy
    output_np = output_binary.cpu().numpy().squeeze()
    l, r = output_np[0,:,:], output_np[1,:,:]
    l, r = mask_repair(l), mask_repair(r)
    df = imageio.imread(df_src)
    l_o, r_o = df*l, df*r
    imageio.imwrite(l_save, l_o)
    imageio.imwrite(r_save, r_o)
    if index == 11:
        # figure
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        imageio.imwrite('l.tif', l_o)
        imageio.imwrite('r.tif', r_o)
        axs[0].imshow(l_o, cmap='gray')
        axs[0].set_title('Left')
        axs[1].imshow(r_o, cmap='gray')
        axs[1].set_title('Right')
        plt.savefig(f'temp_pic/seg_raw.png')


