import os, sys
import os.path as osp
import glob

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import pandas as pd
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle
from sklearn.metrics import average_precision_score
from multiprocessing import Pool
from . import DistEvalHook
from mmdet.utils import check_match, coords2str, expand_df

sys.path.append('/data/ahkamal/6-DoF_Vehicle_Pose_Estimation_Through_Deep_Learning/')
from tools.visualise_pred import visualise_pred

from configs.htc.htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_wudi import work_dir, ds_dir


def match(avg_tr_er):
    # print(*avg_tr_er[0])
    return check_match(*avg_tr_er[0])


class KaggleEvalHook(DistEvalHook):

    def __init__(self, dataset, conf_thresh, interval=1):
        self.ann_file = dataset.ann_file
        self.conf_thresh = conf_thresh

        img_prefix = dataset.img_prefix[:-1] if dataset.img_prefix[-1] == "/" else dataset.img_prefix
        self.dataset_name = os.path.basename(img_prefix)
        print(self.dataset_name)

        super(KaggleEvalHook, self).__init__(dataset, interval)

    def evaluate(self, runner, results):
        
        predictions = {}

        CAR_IDX = 2  # this is the coco car class
        for idx_img, output in enumerate(results):
            # Wudi change the conf to car prediction
            conf = output[0][CAR_IDX][:, -1]  # output [0] is the bbox

            idx = conf > self.conf_thresh 
            
            file_name = os.path.basename(output[2]["file_name"])
            ImageId = ".".join(file_name.split(".")[:-1])

            euler_angle = np.array([quaternion_to_euler_angle(x) for x in output[2]['quaternion_pred']])
            # euler_angle[:, 0],  euler_angle[:, 1], euler_angle[:, 2] = -euler_angle[:, 1], -euler_angle[:, 0], -euler_angle[:, 2]
            translation = output[2]['trans_pred_world']
            coords = np.hstack((euler_angle[idx], translation[idx], conf[idx, None]))
            coords_str = coords2str(coords)
            predictions[ImageId] = coords_str
            
        pred_dict = {'ImageId': [], 'PredictionString': []}
        for k, v in predictions.items():
            pred_dict['ImageId'].append(k)
            pred_dict['PredictionString'].append(v)

        pred_df = pd.DataFrame(data=pred_dict)
        gt_df = pd.read_csv(os.path.join(ds_dir,'_val.csv'))
        expanded_train_df = expand_df(gt_df, ['model_type', 'yaw', 'pitch', 'roll', 'x', 'y', 'z'])

        # get the number of cars
        num_cars_gt = len(expanded_train_df) # total number of cars in val dataset
        ap_list = []

        max_workers = 10 
        p = Pool(processes=max_workers)

        avg_tr_er = []
        max_tr_er = []
        min_tr_er = []

        avg_rot_er = []
        max_rot_er = []
        min_rot_er = []

        avg_score = []

        for result_flg, scores, predicted_tp, mean_tr_error, max_tr_error, min_tr_error, mean_rot_error,\
            max_rot_error, min_rot_error in p.imap(match, zip([(i, gt_df, pred_df) for i in range(1)])): # based on number of thresholds given
            if np.sum(result_flg) > 0:
                n_tp = np.sum(result_flg) # number of true positives detected
                recall = n_tp / num_cars_gt
                
        """
        For single threshold
        """
        if predicted_tp and mean_tr_error and max_tr_error\
        and min_tr_error and mean_rot_error and max_rot_error and min_rot_error:
            print("\nAvg translation error: {}m".format(round(mean_tr_error,3)))
            print("Min T error: {}m - Max T error: {}m\n".format(min_tr_error, max_tr_error))
            print("Avg rotation error: {}deg".format(round(mean_rot_error,3)))
            print("Min R error: {}deg - Max R error: {}deg".format(min_rot_error, max_rot_error))
            print("Avg network confidence (bbox pred): {}%\n".format(round(np.mean(scores),2)*100))
            print('Val {} images mAP is: {}% ({} images)\n'.format(num_cars_gt, round(recall,3)*100,\
                np.floor(num_cars_gt*recall).astype('int')))

        else:
            print('No TP predicted!')
            recall = 0.0

        """
        Records mAP to txt file for every epoch
        (used to plot training loss curve in Tensorboard)
        """
        files_list = glob.glob('{}*'.format(work_dir))
        current_dir = max(files_list,key=os.path.getctime)

        with open(os.path.join(current_dir,'mAP.log'),'a') as file:
            file.write('{}'.format(recall))
            file.write("\n")
        file.close()

        key = 'mAP/{}'.format(self.dataset_name)
        runner.log_buffer.output[key] = recall
        runner.log_buffer.ready = True

