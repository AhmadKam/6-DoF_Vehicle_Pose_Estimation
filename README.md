# 6-DoF Vehicle Pose Estimation through Deep Learning
This repository includes the code used to conduct vehicle pose estimation, as well as the guidelines involved to run it.

## Documentation
* My Capstone thesis can be found [here](docs/ahmad_capstone_thesis.pdf).
* A video of demonstrating how to create a dataset in Blender can be found [here](docs/blender_demo.mp4).
* A video of demonstrating model training and validation can be found [here](docs/training_demo.mp4).
* A video for demonstrating model testing can be found [here](inference_demo.mp4).

## Pre-Trained Models
Pre-trained models can be found [here](https://drive.google.com/drive/folders/1CV5dwUsc1zXNS27ce_a_Q0Hz33TXp1fo?usp=sharing).

Below is a summary of the results obtained with each model:

* Thresholds: Translation: <20cm | Rotation: <5deg
|Scene              | Avg. Translation Error (cm) | Avg. Rotation Error (deg) | mAP |
|:------------------: | :-------:|:-------------------------:|
|360 - 20m          | 12.7  | 0.473 |   49.4% |
|360 - 15m          | 10.1  | 0.468 |   77.8% |
|360 - 10m          | 7.0   | 0.304 |   54.2% |
|Front Left - 10m   | 4.8   | 0.025 |    100% |

* Thresholds: Translation: <2cm | Rotation: <5deg
|Scene              | Avg. Translation Error (cm) | Avg. Rotation Error (deg) | mAP |
|:------------------: | :-------:|:-------------------------:|
|Front Left - 10m   | 1.2   | 0.018 |   15.8% |
|Back Left - 10m    | 1.2   | 0.282 |   12.8% |
|Front Centre - 10m | 1.2   | 0.016 |   17.2% |
|Front Centre - 7m  | 1.2   | 0.016 |   46.4% |
|Front Centre - 5m  | 1.0   | 0.009 |   88.8% |
|Front Centre - 4m  | 1.2   | 0.008 |   12.4% |
|Front Centre - 3m  | 1.5   | 0.009 |    1.8% |

More details can be found in Chapter 5.3 of my thesis.

## Preliminary Steps
### Requirements
The project was implemented with the following hardware and software:
* OS: Red Hat Linux 7.7
* 2x NVIDIA Quadro P4000 (atlas cluster node)
* Python 3.7.7
* CUDA 9.2
* Pytorch 1.4.0
* mmdet-1.0rc0+e104ba2-py3.7-linux-x86_64.egg
* Blender 2.79a

**NOTE**: Instructions for setting up the Anaconda environment can be found [here](docs/INSTALL.md).

### Directories
To successfully run the code, the directories that need to be changed can be found [here](docs/IMPLEMENTATION.md#directories).

## Folder Structure
* Blender Script
The following folder structure is required to run the Blender script. 
```
blender
|   van_model_render.blend
|   get_bbox.py
|   load_annot.py
|   car_models.py
|__ car_models_json
    |   mercedes-vito-van.json
```

* Dataset Images
This directory will be generated automatically by the Blender script.
```
rendered_image
|   all_imgs_coords_points.npz  # contains model vertices pixel coordinates for dataset images
|__ Cam.000                     # this will be the camera's name in Blender
    |   _train.csv
    |   _train.json
    |__ train                   # contains training images
    |
    |   _val.csv
    |   _val.json
    |   _val.txt
    |__ val                     # contains validation images
    |
    |   _test.csv
    |   _test.json
    |   _test.txt
    |__ test                    # contains test images
```

## Implementation
### Running the code
**NOTE**: Make sure to complete the preliminary steps outlined [here](docs/IMPLEMENTATION.md) before running the code.
#### Training
To train the network, run [train_init.sh](train_init.sh) in the Singulary Environment terminal.
This script initialises the Anaconda env and runs [train_kaggle_pku.py](tools/train_kaggle_pku.py), based on the number of GPUs being used. For example:
* Multi-gpu training: `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_kaggle_pku.py --launcher pytorch`
**NOTE**
For single gpu training, use the following: `python train_kaggle_pku.py`. Single gpu training does **NOT** support validation evaluation.

#### Tensorflow Curves
Given data collected during training (e.g. losses, mAP), curves could be plotted showing progression during the learning process:
* Given the log directories, run: `python plot_TFCurves.py` from [tools](tools).
* To visualise the plots, run:`tensorboard --logdir=path_to_event_file`
where *path_to_event_file* is the directory where the event file is created.
then navigate to https://localhost:6006/

#### Inference
To test the network, run [test_init.sh](test_init.sh) in the Singulary Environment terminal.
This script initialises the Anaconda env and runs [test_kaggle_pku.py](tools/test_kaggle_pku.py). 
It will also generate a *.csv* file upon completion.

## Troubleshooting
The following errors may occur when using an iHPC cluster node:
    * Error: `CUDA_HOME variable not found`:
        * Fix: `CUDA_HOME=/usr/local/cuda-9.2 python setup.py install`
    * Error: `dlopen or dlsym: libcaffe2_nvrtc.s RUNTIME_ERROR`
        * Fix: `export LD_LIBRARY_PATH=/home/*ahkamal*/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH`
        **NOTE**: This command will need to be used upon every node login (view [init.sh](init.sh))
    * Error: `FileNotFoundError`
        * Fix: Ensure that [these](docs/IMPLEMENTATION.md#directories) directories have all been correctly set.
    * Use `nvidia-smi` for node GPU information and current processes.
        **NOTE**: To kill a process use: `kill -9 PIDNumber`
