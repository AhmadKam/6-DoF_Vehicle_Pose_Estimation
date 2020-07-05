"""
Script to overwrite original mmdet files 
with the modified ones from the mmdet repository folder.
"""
import os
import argparse
import shutil

def parse_args():
    root_dir = '/home/ahkamal/anaconda3/envs/open-mmlab-2/lib/python3.7/site-packages'

    parser = argparse.ArgumentParser(description='Overwrite files')
    parser.add_argument('--mmdet_path',
                        default=os.path.join(root_dir,'mmdet-1.0rc0+e104ba2-py3.7-linux-x86_64.egg/mmdet/'),
                        help='path to mmdet folder in Anaconda env')
    parser.add_argument('--mmcv_path',
                        default=os.path.join(root_dir,'mmcv/'),
                        help='path to mmcv folder in Anaconda env')
    
    args = parser.parse_args()

    return args

def overwrite_original():
    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    modified_path = os.path.join(repo_path,'mmdet/') # modified files from repo
    
    args = parse_args()

    shutil.copy(os.path.join(modified_path,'apis/train.py'),os.path.join(args.mmdet_path,'apis/'))
    
    shutil.copy(os.path.join(modified_path,'datasets/visualisation_utils.py'),os.path.join(args.mmdet_path,'datasets/'))
    shutil.copy(os.path.join(modified_path,'datasets/kaggle_pku.py'),os.path.join(args.mmdet_path,'datasets/'))
    shutil.copy(os.path.join(modified_path,'datasets/kaggle_pku_utils.py'),os.path.join(args.mmdet_path,'datasets/'))
    
    shutil.copy(os.path.join(modified_path,'datasets/pipelines/transforms.py'),os.path.join(args.mmdet_path,'datasets/pipelines/'))
    shutil.copy(os.path.join(modified_path,'datasets/pipelines/formatting.py'),os.path.join(args.mmdet_path,'datasets/pipelines/'))
    
    shutil.copy(os.path.join(modified_path,'models/detectors/htc.py'),os.path.join(args.mmdet_path,'models/detectors/'))
    shutil.copy(os.path.join(modified_path,'models/bbox_heads/translation_head.py'),os.path.join(args.mmdet_path,'models/bbox_heads/'))
    
    shutil.copy(os.path.join(modified_path,'core/evaluation/kaggle_hooks.py'),os.path.join(args.mmdet_path,'core/evaluation/'))
    
    shutil.copy(os.path.join(modified_path,'utils/map_calculation.py'),os.path.join(args.mmdet_path,'utils/'))

    shutil.copy(os.path.join(repo_path,'tools/runner.py'),os.path.join(args.mmcv_path,'runner/'))