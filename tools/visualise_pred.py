import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from mmcv import imread, imwrite, imshow
from mmdet.datasets.car_models import car_id2name
from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle
from mmdet.datasets.visualisation_utils import draw_box_mesh_kaggle_pku

def visualise_pred(outputs, predicted_tp, ann_file='/home/ahkamal/Desktop/rendered_image/Cam.000/_test.json',
                            outdir='/home/ahkamal/Desktop/'):
        car_cls_coco = 2

        unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                            19, 20, 23, 25, 27, 28, 31, 32,
                            35, 37, 40, 43, 46, 47, 48, 50,
                            51, 54, 56, 60, 61, 66, 70, 71, 76, 79]

        annotations = []
        outfile = ann_file
        if os.path.isfile(outfile):
            annotations = json.load(open(outfile, 'r'))
        
        # annotations = self.clean_corrupted_images(annotations)
        # annotations = self.clean_outliers(annotations) # ADDED - commented

        # self.print_statistics_annotations(annotations)

        for idx in tqdm(range(len(annotations))):
            ann = annotations[idx]
            img_name = ann['filename']
            if img_name.split('/')[-1].split('.png')[0] not in predicted_tp: # ADDED - if statement - only visualise true positives
                continue
            if not os.path.isfile(img_name):
                assert "Image file does not exist!"
            else:
                image = imread(img_name)
                # ADDED - if statement
                if len(outputs) > 3:
                    output = outputs[idx]
                else:
                    output = outputs

                # output is a tuple of three elements
                bboxes, segms, six_dof = output[0], output[1], output[2]
                car_cls_score_pred = six_dof['car_cls_score_pred']
                quaternion_pred = six_dof['quaternion_pred']
                trans_pred_world = six_dof['trans_pred_world']
                euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
                car_labels = np.argmax(car_cls_score_pred, axis=1)
                kaggle_car_labels = [unique_car_mode[x] for x in car_labels]
                car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])

                assert len(bboxes[car_cls_coco]) == len(segms[car_cls_coco]) == len(kaggle_car_labels) \
                       == len(trans_pred_world) == len(euler_angle) == len(car_names)
                # now we start to plot the image from kaggle
                im_combime, iou_flag = visualise_box_mesh(img_name.split('/')[-1][:-4],image, bboxes[car_cls_coco], segms[car_cls_coco],
                                                               car_names, euler_angle, trans_pred_world)
                # plt.imshow(im_combime)
                # plt.show()
                imwrite(im_combime, os.path.join(outdir + '_mes_vis1/' + img_name.split('/')[-1]))

def visualise_box_mesh(name, image, bboxes, segms, car_names, euler_angle, trans_pred_world):

        car_model_dict = load_car_models()

        all_imgs_coords_points = np.load('/home/ahkamal/Desktop/rendered_image/all_imgs_coords_points.npz')

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
                                                        #all_imgs_coords_points,
                                                        trans_pred_world,
                                                        euler_angle)

        return im_combime, iou_flag

def load_car_models(outdir='/data/ahkamal/Kaggle_PKU_Baidu_v1/data/Kaggle/pku-autonomous-driving/'):
        car_model_dir = os.path.join(outdir, 'car_models_json')
        car_model_dict = {}
        for car_name in tqdm(os.listdir(car_model_dir)):
            with open(os.path.join(outdir, 'car_models_json', car_name)) as json_file:
                car_model_dict[car_name[:-5]] = json.load(json_file)

        return car_model_dict