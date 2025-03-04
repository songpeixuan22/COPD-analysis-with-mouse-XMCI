#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loader_mask.py
~~~~~~~~~~~~~~~~~~~~~
This script uses .tif images to load the mask for unet training.

:author: Peixuan Song
:email: spx22@mails.tsinghua.edu.cn
:copyright: (c) 2025 by Peoxuan Song.
:license: MIT License, see LICENSE for more details.
"""

__version__ = "0.1.0"

import os
import imageio
import numpy as np

# set the source and destination folder
src_folder = 'data_groundtruth'
dst_folder = 'dataset/mask'

# make sure the destination folder exists
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)
# make the left and right subfolders
l_dst_path = os.path.join(dst_folder, 'l')
r_dst_path = os.path.join(dst_folder, 'r')
if not os.path.exists(l_dst_path):
    os.makedirs(l_dst_path, exist_ok=True)
    os.makedirs(r_dst_path, exist_ok=True)

# process each subfolder
for subfolder in os.listdir(src_folder):
    subfolder_path = os.path.join(src_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # read the left and right tif files
        l_tif_path = os.path.join(subfolder_path, 'l.tif')
        r_tif_path = os.path.join(subfolder_path, 'r.tif')
        l_image = imageio.imread(l_tif_path)
        r_image = imageio.imread(r_tif_path)

        # clip the images
        for i in range(l_image.shape[0]):
            l_slice = l_image[i]
            r_slice = r_image[i]

            # convert the non-zero values to 1
            l_slice = np.where(l_slice != 0, 1, l_slice)
            r_slice = np.where(r_slice != 0, 1, r_slice)

            # save the slices
            l_slice_name = f"{subfolder}_slice_{i}.tif"
            r_slice_name = f"{subfolder}_slice_{i}.tif"
            l_slice_path = os.path.join(l_dst_path, l_slice_name)
            r_slice_path = os.path.join(r_dst_path, r_slice_name)
            imageio.imwrite(l_slice_path, l_slice)
            imageio.imwrite(r_slice_path, r_slice)
