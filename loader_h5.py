#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loader_h5.py
~~~~~~~~~~~~~~~~~~~~~
This script select the needed slices from .hdf5 files,
and save them as .tif files (you can make your own version!).

:author: Peixuan Song
:email: spx22@mails.tsinghua.edu.cn
:copyright: (c) 2025 by Peixuan Song.
:license: MIT License, see LICENSE for more details.
"""

__version__ = "0.1.0"

import h5py
import os
import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# if 3d
three = 1
# if 3d, the slices on the 2d axis to be selected
seg_dic = {
    '241215_403':[88,202,665,805],
    '241222_403':[113,170,690,730],
    '250105_403':[172,228,765,808],
    '250112_403':[189,209,770,780],
    '250118_403':[196,217,733,798],
    '250216_403':[197,170,721,728],
    '250112_482':[146,269,733,832],
    '250118_482':[104,213,669,757],
    '250216_482':[176,237,696,817]
}

# set the temp path
pic_path = 'temp_pic/'
npy_path = 'temp_npy/'
csv_path = 'temp_csv/'
# set the folder name
folder = '250216_482'
# set the raw data path
raw_path = f'data_h5/{folder}.hdf5' 
abs_p = f'data_tif/{folder}/abs'
df_p = f'data_tif/{folder}/df'
os.makedirs(abs_p, exist_ok=True)
os.makedirs(df_p, exist_ok=True)
# set the slices to be selected
slices = list(np.arange(200, 350, 1))


if three == 1:
    abss = []
    dff = []
    with h5py.File(raw_path, 'r') as f:
        keys = list(f.keys())
        [y0, x0, y1, x1] = seg_dic[folder]
        abs = f[keys[0]][:,x0:x1,y0:y1]
        df = f[keys[1]][:,x0:x1,y0:y1]
        for i, s in tqdm(enumerate(slices), total=len(slices)):
            img_df = df[s]  # clip the slices
            img_abs = abs[s]
            abss.append(img_abs)
            dff.append(img_df)
        abss = np.stack(abss, axis=0)
        dff = np.stack(dff, axis=0)
        imageio.mimwrite(f'{abs_p}/slice_t.tif', abss)
        imageio.mimwrite(f'{df_p}/slice_t.tif', dff)
        breakpoint()
else:
    with h5py.File(raw_path, 'r') as f:
        keys = list(f.keys())
        abs = f[keys[0]]
        df = f[keys[1]]
        for i, s in tqdm(enumerate(slices), total=len(slices)):
            img_df = df[s]  # clip the slices
            img_abs = abs[s]
            # save the slices
            imageio.imwrite(f'{abs_p}/slice_{i}.tif', img_abs)
            imageio.imwrite(f'{df_p}/slice_{i}.tif', img_df)

