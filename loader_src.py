#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loader_src.py
~~~~~~~~~~~~~~~~~~~~~
This script uses .tif images to load the input for unet training.

:author: Peixuan Song
:email: spx22@mails.tsinghua.edu.cn
:copyright: (c) 2025 by Peixuan Song.
:license: MIT License, see LICENSE for more details.
"""

__version__ = "0.1.0"

import os
import numpy as np
import imageio

# set the source and destination folder
src_folder = 'data_tif'
dst_folder = 'dataset/raw'

rewrite = 1

# make sure the destination folder exists
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# process each subfolder
for subfolder in os.listdir(src_folder):
    subfolder_path = os.path.join(src_folder, subfolder)
    if os.path.isdir(subfolder_path):
        abs_path = os.path.join(subfolder_path, 'abs')
        if os.path.isdir(abs_path):
            # process each tif file
            for filename in os.listdir(abs_path):
                if filename.endswith('.tif'):
                    # rename the file
                    new_filename = f"{subfolder}_{filename}"
                    dstfilepath = os.path.join(dst_folder, new_filename)
                    if rewrite != 1 and os.path.exists(dstfilepath):
                        break
                    else:
                        # read the tif file
                        srcfilepath = os.path.join(abs_path, filename)
                        img = imageio.imread(srcfilepath)
                        img = np.where(img <= 0.03, 1, img)
                        # save the file
                        imageio.imwrite(dstfilepath, img)
                        print(f"Saved {dstfilepath}")
