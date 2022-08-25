# SunnyNet
Rain and fog point cloud filtering network.

We use `PyTorch` to implement a CNN to filter out rain and fog point clouds. This approach is mainly based on our paper `"Semantic Segmentation Based Rain and Fog Filtering Only by LiDAR Point Clouds"` published at the 2022 IEEE International Conference on Unmanned Systems (ICUS).


# Installation
This software has only been tested on `ubuntu 18.04(x64)`, `torch==1.8.1`. Other version of PyTorch has not been tested. To install the required packages you may run
```
pip install -r requirements.txt
```


# Train your model
## Data preparation
As described in the paper, we use the adverse weather dataset from `DENSE`. You can click [here](https://www.uni-ulm.de/index.php?id=101568) for registration and download. Thanks for the contribution of the DENSE dataset to the point cloud segmentation task under adverse weather.
The structure of the folder which storing datasets should be as follows, including three subfolders: `train_ 01`, `test_ 01`, `val_ 01`.
```
dataset
|——train_01
|  |——2018-11-28_104809_Static4-Clear
|  |  |——LidarImage_000000001.hdf5
|  |  |——LidarImage_000000002.hdf5
|  |  ...
|  |——2018-11-28_112851_Static4-FogA
|  ...
|
|——test_01
|  |
|  ...
|
|——val_01
   |
   ...
```

## Start training
After downloading the data set, you may call the following commands on the terminal to train your model.
```
python main_dense.py --attention-type eca --epochs 30 --dataset-dir /PATH/TO/YOUR/DATASET
```
For `--attention-type`, you can select 4 modes: `cbam`, `eca`, `senet` and `original`. You can found the meaning of other parameters in `main_dense.py`.


# Test model


