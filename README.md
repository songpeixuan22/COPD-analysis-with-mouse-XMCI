# COPD analysis with mouse XMCI

detailed description and pipeline to be added.

<!-- 文件说明：
data_raw: 完全没有处理过的重建后文件，格式为hdf5；
data_unprocessed: 选取的hdf5的切片，未分割肺部，格式为tif；
data_processed: 已经切割肺部后的切片，格式为3Dtif；
data: 直接用于分析的左右肺tif切片；
temp_csv: 分析过程中的csv文件；
temp_npy: 分析过程中的npy文件；
temp_pic: 分析过程中的图片；

data_divide.py: 处理左右肺在同一张图片的情况；
data_process.py: 处理图片生成特征向量csv文件；
find_rect.py: 进一步分割单个肺；
loader_*.py: 加载各种文件；
data_divide.py: 
data_divide.py: 
data_divide.py:  -->