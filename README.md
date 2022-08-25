# SunnyNet
## Rain and fog point cloud filtering network.
We use `PyTorch` to implement a CNN to filter out rain and fog point clouds. This approach is mainly based on our paper `"Semantic Segmentation Based Rain and Fog Filtering Only by LiDAR Point Clouds"` published at the 2022 IEEE International Conference on Unmanned Systems (ICUS).


# Installation
This software has only been tested on `ubuntu 18.04(x64)`, `torch==1.8.1`. Other version of PyTorch has not been tested. To install the required packages you may call the following commands.
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

## Train from checkpoint
If you want to continue training from the checkpoint, you may call the following commands.
```
python main_dense.py --attention-type eca --epochs THE-EPOCH-YOU-WANT --resume /PATH/TO/YOUR/CHECKPOINT --dataset-dir /PATH/TO/YOUR/DATASET
```
Note that checkpoint and model are different files. ***When testing, you should load the model file. When continuing training, you should load the checkpoint file.***


# Test model
If you want to use the model for testing, you may call the following commands. The terminal will output test results. We have provided a pre-training model based on eca, which is located in `./checkpoints/model_eca_2022-08-24_15_30_56/model_epoch3_mIoU=89.7.pth`.
```
python main_dense.py --attention-type eca --resume /PATH/TO/YOUR/MODEL --dataset-dir /PATH/TO/YOUR/DATASET --test True
```


# Test LROR
We provide LROR as a post-processing step in our network. Please read our paper to understand its specific principle. To test the effect of lror, you may call the following commands.
```
python test_LROR.py --attention-type eca --model-path /PATH/TO/YOUR/MODEL --after-processing lror --remove-zero True
```
The `--remove-zero` parameter is used to remove the point clouds with XYZ coordinates of 0 in the DENSE dataset to reduce unnecessary calculation cost.


# Using SunnyNet under ROS
Create your workspace
```
mkdir -p catkin_ws/src
cd catkin_ws/src
catkin_init_workspace
```
Put the software package `point_cloud_denoising` provided by us under `catkin_ws/src`, then compile the workspace.
```
cd catkin_ws
catkin_make
source devel/setup.bash
```
To display the HDF5 point cloud files，you may call the following commands.
```
roslaunch point_cloud_denoising display.launch
```
To convert HDF5 point cloud files into 2D depth maps，you may call the following commands.
```
roslaunch point_cloud_denoising lidar2img.launch
```
To use SunnyNet under the ROS framework，you may call the following commands.
```
roslaunch point_cloud_denoising point_cloud_denoising.launch
```
If you want to use SunnyNet from HDF5 point cloud files under the ROS framework，you need to open `point_ cloud_ denoising.launch` and revise `<node name="point_cloud_denoising" pkg="point_cloud_denoising" type="ros_denoising.py" output="screen"/>` to `<node name="point_cloud_denoising" pkg="point_cloud_denoising" type="ros_denoising_from_hdf5.py" output="screen"/>`.
