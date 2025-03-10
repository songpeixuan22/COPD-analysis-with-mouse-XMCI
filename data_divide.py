#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_divide.py
~~~~~~~~~~~~~~~~~~~~~
This script uses the 2d slices generated by unet to get left&right lungs precisely, 
without mach black background (you can write you own version!).

:author: Lei Hu
:email: hul22@mails.tsinghua.edu.cn
:copyright: (c) 2025 by Lei Hu.
:license: MIT License, see LICENSE for more details.
"""

__version__ = "0.1.0"



import imageio.v2 as imageio
import numpy as np
import os
import shutil
import glob


def crop_nonzero_boundary(arr):
    # get the row and column indices of all non-zero elements in the array
    nonzero_rows = np.any(arr != 0, axis=1)  # whether each row has non-zero elements
    nonzero_cols = np.any(arr != 0, axis=0)  # whether each column has non-zero elements

    # find the top, bottom, left, and right non-zero boundaries
    top = np.argmax(nonzero_rows)
    bottom = arr.shape[0] - np.argmax(nonzero_rows[::-1]) - 1
    left = np.argmax(nonzero_cols)
    right = arr.shape[1] - np.argmax(nonzero_cols[::-1]) - 1

    # crop the subarray containing non-zero elements
    cropped_arr = arr[top : bottom + 1, left : right + 1]

    return cropped_arr


# create folders
base_dir = "data_processed"

left_dir_normal = os.path.join(base_dir, "left_normal")
right_dir_normal = os.path.join(base_dir, "right_normal")
left_dir_abnormal = os.path.join(base_dir, "left_abnormal")
right_dir_abnormal = os.path.join(base_dir, "right_abnormal")

# if the base_dir exists, delete it and all its subfolders and files
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
    print(f" {base_dir} folder has been deleted.")
else:
    print(f" {base_dir} folder does not exist.")

# if the base_dir does not exist, create it
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    print(f"'{base_dir}' folder has been created.")

if not os.path.exists(left_dir_normal):
    os.makedirs(left_dir_normal)
    print(f"'{left_dir_normal}' folder has been created.")
if not os.path.exists(right_dir_normal):
    os.makedirs(right_dir_normal)
    print(f"'{right_dir_normal}' folder has been created.")
if not os.path.exists(left_dir_abnormal):
    os.makedirs(left_dir_abnormal)
    print(f"'{left_dir_normal}' folder has been created.")
if not os.path.exists(right_dir_abnormal):
    os.makedirs(right_dir_abnormal)
    print(f"'{right_dir_abnormal}' folder has been created.")

# get tif files
tif_left_normal = [
    # "250118_401/250118_402",
    "250216_403/4826",
    # "250216_403/250118_425",
    "250216_403/250118_482",
    # "250216_403/250216_402",
]
prefix_left_normal = "data_unprocessed/"
postfix_left_normal = "/l.tif"
tif_left_normal = list(
    map(lambda x: prefix_left_normal + x + postfix_left_normal, tif_left_normal)
)

tif_right_normal = [
    # "250118_401/250118_402",
    "250216_403/4256",
    # "250216_403/250118_425",
    "250216_403/250118_482",
    # "250216_403/250216_402",
]
prefix_right_normal = "data_unprocessed/"
postfix_right_normal = "/r.tif"
tif_right_normal = list(
    map(lambda x: prefix_right_normal + x + postfix_right_normal, tif_right_normal)
)

tif_left_abnormal = [
    # "250118_401/250118_401",
    "250216_403/250118_403",
    # "250216_403/250216_401",
    "250216_403/250216_403",
]
prefix_left_abnormal = "data_unprocessed/"
postfix_left_abnormal = "/l.tif"
tif_left_abnormal = list(
    map(lambda x: prefix_left_abnormal + x + postfix_left_abnormal, tif_left_abnormal)
)

tif_right_abnormal = [
    # "250118_401/250118_401",
    "250216_403/250118_403",
    # "250216_403/250216_401",
    "250216_403/250216_403",
]
prefix_right_abnormal = "data_unprocessed/"
postfix_right_abnormal = "/r.tif"
tif_right_abnormal = list(
    map(
        lambda x: prefix_right_abnormal + x + postfix_right_abnormal, tif_right_abnormal
    )
)

left_num = 0
for file in tif_left_normal:
    img = imageio.imread(file)

    for i in range(img.shape[0]):
        left_path_normal = f"{left_dir_normal}/left_part{left_num}.tif"
        left_num += 1
        left_i = img[i, :, :]
        imageio.imwrite(left_path_normal, crop_nonzero_boundary(left_i))

right_num = 0
for file in tif_right_normal:
    img = imageio.imread(file)

    for i in range(img.shape[0]):
        right_path_normal = f"{right_dir_normal}/right_part{right_num}.tif"
        right_num += 1
        right_i = img[i, :, :]
        imageio.imwrite(right_path_normal, crop_nonzero_boundary(right_i))

# read 2d tif files
tif_left = [
    "data_unprocessed/abnormal/241215_403/l/*.tif",
    "data_unprocessed/abnormal/241222_403/l/*.tif",
    # "data_unprocessed/abnormal/241229_403/l/*.tif",
    "data_unprocessed/abnormal/250105_403/l/*.tif",
    "data_unprocessed/abnormal/250112_403/l/*.tif",
]

tif_right = [
    "data_unprocessed/abnormal/241215_403/r/*.tif",
    "data_unprocessed/abnormal/241222_403/r/*.tif",
    # "data_unprocessed/abnormal/241229_403/r/*.tif",
    "data_unprocessed/abnormal/250105_403/r/*.tif",
    "data_unprocessed/abnormal/250112_403/r/*.tif",
]

# read all tif files
left_num = 0
for file_path in tif_left:
    left_tif = sorted(glob.glob(file_path))
    for file in left_tif:
        img = imageio.imread(file)
        left_path = os.path.join(left_dir_abnormal,f"left_part{left_num}.tif")
        left_num += 1
        imageio.imwrite(left_path, crop_nonzero_boundary(img))


right_num = 0
for file_path in tif_right:
    right_tif = sorted(glob.glob(file_path))
    for file in right_tif:
        img = imageio.imread(file)
        right_path = f"{right_dir_abnormal}/right_part{right_num}.tif"
        right_num += 1
        imageio.imwrite(right_path, crop_nonzero_boundary(img))


for file in tif_left_abnormal:
    img = imageio.imread(file)

    for i in range(img.shape[0]):
        left_path_abnormal = f"{left_dir_abnormal}/left_part{left_num}.tif"
        left_num += 1
        left_i = img[i, :, :]
        imageio.imwrite(left_path_abnormal, crop_nonzero_boundary(left_i))

for file in tif_right_abnormal:
    img = imageio.imread(file)

    for i in range(img.shape[0]):
        right_path_abnormal = f"{right_dir_abnormal}/right_part{right_num}.tif"
        right_num += 1
        right_i = img[i, :, :]
        imageio.imwrite(right_path_abnormal, crop_nonzero_boundary(right_i))
