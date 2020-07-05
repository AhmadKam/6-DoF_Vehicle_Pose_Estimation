import multiprocessing
multiprocessing.set_start_method('spawn', True)

import argparse
import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import os.path as osp
import shutil
import tempfile
import pandas as pd
import numpy as np
import timeit, time
from numpy import linalg as LA
from math import sqrt, log
import cv2

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint, Runner
from mmcv import Config

from mmdet.apis import init_dist
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils.map_calculation import TranslationDistance, RotationDistance, check_match
from mmdet.apis.train import batch_processor, build_optimizer

from mmdet.datasets.kaggle_pku_utils import euler_to_Rot, quaternion_to_euler_angle, filter_igore_masked_using_RT, filter_output
from tqdm import tqdm
from multiprocessing import Pool

from visualise_pred import visualise_pred
from sklearn.metrics import average_precision_score

from configs.htc.htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_wudi import ds_dir

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    args = parse_args()
    cfg = Config.fromfile(args.config)
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir, is_test=True)

    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = runner.model(return_loss=False, rescale=True, **data)
        results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def write_submission(estimations, args, dataset,
                     conf_thresh=0.8,
                     filter_mask=False,
                     horizontal_flip=False,
                     plot_tp=False,
                     save_tp=False):
    
    img_prefix = dataset.img_prefix
    submission = args.out.replace('.pkl', '')
    submission += '_' + img_prefix.split('/')[-1]
    submission += '_conf_' + str(conf_thresh)

    if filter_mask:
        submission += '_filter_mask'
    elif horizontal_flip:
        submission += '_horizontal_flip'

    submission += '.csv'
    predictions = {}


    CAR_IDX = 2  # this is the coco car class

    for img_name in estimations:
        for idx_img, output in tqdm(enumerate(estimations[img_name])):
            file_name = os.path.basename(output[2]["file_name"])
            ImageId = ".".join(file_name.split(".")[:-1])

            # Wudi change the conf to car prediction
            if len(output[0][CAR_IDX]):
                conf = output[0][CAR_IDX][:, -1]  # output [0] is the bbox
                """
                Bbox check
                """               
                idx_conf = conf > conf_thresh
                if filter_mask:
                    # this filtering step will takes 2 second per iterations
                    idx_keep_mask = filter_igore_masked_using_RT(ImageId, output[2], img_prefix, dataset)

                    # the final id should require both
                    idx = idx_conf * idx_keep_mask
                else:
                    idx = idx_conf
                # if 'euler_angle' in output[2].keys():
                if False:  # NMR has problem saving 'euler angle' Its
                    eular_angle = output[2]['euler_angle']
                else:
                    eular_angle = np.array([quaternion_to_euler_angle(x) for x in output[2]['quaternion_pred']])
                translation = output[2]['trans_pred_world']
                coords = np.hstack((eular_angle[idx], translation[idx], conf[idx, None]))

                coords_str = coords2str(coords)

                predictions[ImageId] = coords_str
            else:
                predictions[ImageId] = ""

    pred_dict = {'ImageId': [], 'PredictionString': []}
    for k, v in predictions.items():
        pred_dict['ImageId'].append(k)
        pred_dict['PredictionString'].append(v)

    
    ann_file = os.path.join(ds_dir, '_test.csv')
    test_gt_annot = pd.read_csv(ann_file)

    num_cars_gt = len(pred_dict['ImageId'])  

    """
    For single threshold
    """
    result_flg, scores, predicted_tp, mean_tr_error, max_tr_error, \
    min_tr_error, mean_rot_error, max_rot_error, min_rot_error = check_match(0,test_gt_annot, pred_dict)
        
    if np.sum(result_flg) > 0:
        n_tp = np.sum(result_flg) # number of true positives detected
        recall = n_tp / num_cars_gt    
    
    """
    Only writes TP predictions to csv file (Comment this section to write all predictions)
    """
    temp = {'ImageId':[],'PredictionString':[]}
    for i, img in enumerate(pred_dict['ImageId']):
        if img in predicted_tp:
            temp['ImageId'].append(pred_dict['ImageId'][i])
            temp['PredictionString'].append(pred_dict['PredictionString'][i])
    pred_dict = temp


    df = pd.DataFrame(data=pred_dict)
    print("Writing submission csv file to: %s" % submission)
    df.to_csv(submission, index=False)

    if mean_tr_error and min_tr_error and max_tr_error\
    and mean_rot_error and min_rot_error and max_rot_error and scores and recall:
        print("\nAvg translation error: {}m".format(round(mean_tr_error,3)))
        print("Min T error: {}m - Max T error: {}m\n".format(min_tr_error, max_tr_error))
        print("Avg rotation error: {}deg".format(round(mean_rot_error,3)))
        print("Min R error: {}deg - Max R error: {}deg".format(min_rot_error, max_rot_error))
        print("Avg network confidence (bbox pred): {}%\n".format(round(np.mean(scores),2)*100))
        print('Test {} images mAP is: {}% ({} images)\n'.format(num_cars_gt, round(recall,3)*100, len(pred_dict['ImageId'])))

    if (plot_tp or save_tp) and predicted_tp:
        for img_name in estimations:
            visualise_pred(estimations[img_name], predicted_tp, plot=plot_tp,save_img=save_tp)

    return submission


def filter_output_pool(t):
    return filter_output(*t) 


def write_submission_pool(outputs, args, dataset,
                          conf_thresh=0.8,
                          horizontal_flip=False,
                          max_workers=20):
    """
    For accelerating filter image
    :param outputs:
    :param args:
    :param dataset:
    :param conf_thresh:
    :param horizontal_flip:
    :param max_workers:
    :return:
    """
    img_prefix = dataset.img_prefix

    submission = args.out.replace('.pkl', '')
    submission += '_' + img_prefix.split('/')[-1]
    submission += '_conf_' + str(conf_thresh)
    submission += '_filter_mask.csv'
    if horizontal_flip:
        submission += '_horizontal_flip'
    submission += '.csv'
    predictions = {}

    p = Pool(processes=max_workers)
    for coords_str, ImageId in p.imap(filter_output_pool,
                                      [(i, outputs, conf_thresh, img_prefix, dataset) for i in range(len(outputs))]):
        predictions[ImageId] = coords_str

    pred_dict = {'ImageId': [], 'PredictionString': []}
    for k, v in predictions.items():
        pred_dict['ImageId'].append(k)
        pred_dict['PredictionString'].append(v)

    df = pd.DataFrame(data=pred_dict)
    print("Writing submission csv file to: %s" % submission)
    df.to_csv(submission, index=False)
    return submission


def coords2str(coords):
    s = []
    for c in coords:
        for l in c:
            s.append('%.5f' % l)
    return ' '.join(s)


def parse_args():
    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(repo_path,'configs/htc/')

    checkpoint_path = '/data/ahkamal/output_data/Jun14-22-45(frLeft-10m)/epoch_15.pth'
    
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config',
                        default=os.path.join(config_path,'htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_wudi.py'),
                        help='train config file path')
    parser.add_argument('--checkpoint', default=checkpoint_path, help='checkpoint file')
    parser.add_argument('--conf', default=0.9, help='Confidence threshold for writing submission')
    parser.add_argument('--json_out', help='output result file name without extension', type=str)
    parser.add_argument('--eval', type=str, nargs='+',
                        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints', ' kaggle'],
                        help='eval types')
    parser.add_argument('--show', default=False, action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--horizontal_flip', default=False, action='store_true')
    parser.add_argument('--world_size', default=8)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main(plot=False,save_img=False):
    args = parse_args()

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)

    # Wudi change the args.out directly related to the model checkpoint file data
    if args.horizontal_flip:
        args.out = os.path.join(cfg.work_dir,
                                cfg.data.test.img_prefix.split('/')[-2].replace('_images', '_') +
                                args.checkpoint.split('/')[-1][:-4] + '_horizontal_flip.pkl')
        print('horizontal_flip activated')
    else:
        args.out = os.path.join(cfg.work_dir,'inference/{}_{}/'.format(args.checkpoint.split('/')[-2],\
                    args.checkpoint.split('/')[-1].split('.pth')[0]), 
                    args.checkpoint.split('/')[-2 ]+'_'+\
                    args.checkpoint.split('/')[-1].split('.pth')[0]+'.pkl')

        # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader

    dataset = build_dataset(cfg.data.test)
    
    if not os.path.exists(args.out):
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        
        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
     
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility

        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        
        if not distributed:
            estimations = {}
            inference_dir = cfg.work_dir + 'inference/{}_{}/'.format(args.checkpoint.split('/')[-2],\
                            args.checkpoint.split('/')[-1].split('.pth')[0])
            pred_imgs_log = inference_dir + 'predicted_images.log'
                            
            if not os.path.exists(inference_dir):
                os.makedirs(inference_dir)

            if os.path.exists(pred_imgs_log):
                os.remove(pred_imgs_log)
            
            with open(pred_imgs_log,'w'):
                pass

            model = MMDataParallel(model, device_ids=[0])
            input('Press Enter to start pose estimation. (Ctrl+C to stop)\n')
            try:
                while True:
                    dataset = build_dataset(cfg.data.test)    
                    data_loader = build_dataloader(
                        dataset,
                        imgs_per_gpu=1,
                        workers_per_gpu=cfg.data.workers_per_gpu,
                        dist=distributed,
                        shuffle=False)

                    dataloader_name = data_loader._index_sampler.sampler.data_source.img_infos
                    temp = dataloader_name[:]
                    if os.path.exists(pred_imgs_log):
                        with open(pred_imgs_log,'r') as file:
                            pred_imgs = file.readlines()
                        file.close()
                        i = 0
                        while i < len(dataloader_name):
                            if (dataloader_name[i]['filename'].split('/')[-1] + '\n') in pred_imgs:
                                temp.pop(i)
                                i = 0
                            else:
                                break
                                    
                            dataloader_name = temp[:]

                    # Check for new images
                    if len(dataloader_name) == 0:
                        print("\nPose estimation finished.\n")
                        break
                
                    outputs = single_gpu_test(model, data_loader, args.show)
        
                    for j in range(len(outputs)):
                        img_name = outputs[j][2]['file_name'].split('/')[-1] 
                        if len(outputs) == 1:  
                            estimations[img_name] = outputs
                        elif len(outputs) > 1 and j == 0:
                            estimations['all_imgs'] = outputs
                        
                        if plot or save_img:
                            visualise_pred(outputs, plot=plot, save_img=save_img)
                        
                        with open(pred_imgs_log,'a+') as file:
                            pred_imgs = file.readlines()
                            if img_name not in pred_imgs:
                                file.write(img_name + '\n')
                        file.close()

                    # time.sleep(2) # wait until new image is taken
            except KeyboardInterrupt:
                pass
      
        mmcv.dump(estimations, args.out)
    
    else:
        estimations = mmcv.load(args.out)

    if distributed:
        rank, _ = get_dist_info()
        if rank != 0:
            return

    if cfg.write_submission:
        submission = write_submission(estimations, args, dataset,
                         conf_thresh=0.8, # bbox threshold
                         filter_mask=False,
                         horizontal_flip=args.horizontal_flip,
                         plot_tp=False, # plots TP predictions
                         save_tp=False) # saves TP predictions


if __name__ == '__main__':
    main(plot=False,save_img=False) # plot: plots all predictions | save_img: saves all predictions