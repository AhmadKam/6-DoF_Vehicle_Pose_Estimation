#!/bin/sh
# System Initialisation for model testing
source /home/ahkamal/anaconda3/bin/activate

conda activate open-mmlab-2

cd /data/ahkamal/6-DoF_Vehicle_Pose_Estimation_Through_Deep_Learning/tools

export LD_LIBRARY_PATH=/home/ahkamal/anaconda3/envs/open-mmlab-2/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH

python test_kaggle_pku.py