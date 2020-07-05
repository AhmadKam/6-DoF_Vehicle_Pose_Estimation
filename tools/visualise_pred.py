import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import cv2

from mmcv import imread, imwrite, imshow
from mmdet.datasets.car_models import car_id2name
from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle
from mmdet.datasets.visualisation_utils import draw_box_mesh_kaggle_pku

from configs.htc.htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_wudi import ds_dir, load_from


def visualise_pred(outputs, predicted_tp=None, plot=False, save_img=False,ann_file='{}/_test.json'.format(ds_dir),
                            outdir='{}/'.format(os.path.dirname(os.path.dirname(ds_dir)))):
        car_cls_coco = 2

        # unique_car_mode = [79]
        unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                                19, 20, 23, 25, 27, 28, 31, 32,
                                35, 37, 40, 43, 46, 47, 48, 50,
                            51, 54, 56, 60, 61, 66, 70, 71, 76, 79]

        annotations = []
        outfile = ann_file
        if os.path.isfile(outfile):
            annotations = json.load(open(outfile, 'r'))

        for idx in tqdm(range(len(annotations))):
            img_name = outputs[idx][2]['file_name']
            if predicted_tp and img_name.split('/')[-1].split('.png')[0] not in predicted_tp:
                continue
            if not os.path.isfile(img_name):
                assert "Image file does not exist!"
            else:
                image = imread(img_name)
                # image = imread('/home/ahkamal/Desktop/rendered_images/Cam.000/test/van_X0.016_Y0.029_R0.7.png') # added
                if len(outputs) > 3:
                    output = outputs[idx]
                else:
                    output = outputs[0]
                    
                # output is a tuple of three elements
                bboxes, segms, six_dof = output[0], output[1], output[2]
                car_cls_score_pred = six_dof['car_cls_score_pred']
                quaternion_pred = six_dof['quaternion_pred']
                trans_pred_cam = six_dof['trans_pred_world'] # TODO: change name to six_dof['trans_pred_cam']
                euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
                car_labels = np.argmax(car_cls_score_pred, axis=1)
                kaggle_car_labels = [unique_car_mode[x] for x in car_labels]
                car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])

                assert len(bboxes[car_cls_coco]) == len(segms[car_cls_coco]) == len(kaggle_car_labels) \
                       == len(trans_pred_cam) == len(euler_angle) == len(car_names)
                # now we start to plot the image from kaggle
                im_combime, iou_flag = visualise_box_mesh(img_name.split('/')[-1][:-4],image, bboxes[car_cls_coco], segms[car_cls_coco],
                                                               car_names, euler_angle, trans_pred_cam)
                
                
                if plot:
                    plt.imshow(im_combime)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show(block=False)
                    plt.pause(3)
                    plt.close()
                
                if save_img:
                    imwrite(im_combime, os.path.join(outdir + '_mes_vis/' + img_name.split('/')[-1]))
                    
                if len(outputs) != len(annotations): # for real-time plotting
                        break

def visualise_box_mesh(name, image, bboxes, segms, car_names, euler_angle, trans_pred_cam):

        car_model_dict = load_car_models()

        all_imgs_coords_points = np.load('{}/all_imgs_coords_points.npz'.format(os.path.dirname(ds_dir)))
        
        camera_matrix = np.array([[3701.25, 0, 1692.0], 
                                    [0, 2391.6667, 615.0],
                                    [0, 0, 1]], dtype=np.float32) 

        segms[0]['counts'] = segms[0]['counts'].decode("utf-8")
        
        im_combime, iou_flag = draw_box_mesh_kaggle_pku(name,image,
                                                        bboxes,
                                                        segms,
                                                        car_names,
                                                        car_model_dict,
                                                        camera_matrix,
                                                        all_imgs_coords_points,
                                                        trans_pred_cam,
                                                        euler_angle)

        return im_combime, iou_flag

def load_car_models(outdir='{}/6-DoF_Vehicle_Pose_Estimation_Through_Deep_Learning/blender/'\
                    .format(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(load_from)))))):
        car_model_dir = os.path.join(outdir, 'car_models_json')
        car_model_dict = {}
        for car_name in os.listdir(car_model_dir): # TODO: For more than one model, use tqdm for progress bar
            with open(os.path.join(outdir, 'car_models_json', car_name)) as json_file:
                car_model_dict[car_name[:-5]] = json.load(json_file)

        return car_model_dict