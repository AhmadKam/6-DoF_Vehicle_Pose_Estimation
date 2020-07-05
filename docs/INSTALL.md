## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+ (Python 2 is not supported)
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC(G++) 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv) 0.2.10

### Install mmdetection

a. Create a conda virtual environment and activate it (**NOTE**: replace *ahkamal* by your username).

```shell
source /home/ahkamal/anaconda3/bin/activate
conda create -p /home/ahkamal/anaconda3/envs/open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch stable and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

c. Install mmdetection (other dependencies will be installed automatically).

```shell
python setup.py develop
```

d. Run this script from the [tools](tools) folder to overwrite the required files:
```
python overwrite_original.py --mmdet_path=path_to_mmdet --mmcv_path=path_to_mmcv
```
* *path_to_mmdet* corresponds to the path to the mmdet folder in the Anaconda env
* *path_to_mmcv* corresponds to the path to the mmcv folder in the Anaconda env

**Note**:

1. The git commit id will be written to the version number with step c, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, mmdetection is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. The newest mmdet 1.4+ has different API in calling mmcv. Hence, we would recommend installing the mmdet from the uploaded files using: `python setup.py install`)

More information can be found [here](https://github.com/stevenwudi/Kaggle_PKU_Baidu), and [here](https://github.com/stevenwudi/Kaggle_PKU_Baidu).
