"""
Generates json file holding entire dataset annotations
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from math import sin, cos, acos, pi
from mmcv import imread, imwrite
from mmdet.datasets.car_models import car_id2name
from pycocotools import mask as maskUtils
from scipy.spatial.transform import Rotation as R
from mmdet.datasets.kaggle_pku_utils import euler_to_Rot, euler_angles_to_quaternions, \
    quaternion_upper_hemispher, quaternion_to_euler_angle

def RotationDistance(p, g):
	true = [g[1], g[0], g[2]]
	pred = [p[1], p[0], p[2]]
	q1 = R.from_euler('xyz', true)
	q2 = R.from_euler('xyz', pred)
	diff = R.inv(q2) * q1
	W = np.clip(diff.as_quat()[-1], -1., 1.)

	# in the official metrics code:
	# Peking University/Baidu - Autonomous Driving
	#   return Object3D.RadianToDegree( Math.Acos(diff.W) )
	# this code treat θ and θ+2π differntly.
	# So this should be fixed as follows.
	W = (acos(W) * 360) / pi
	if W > 180:
		W = 360 - W
	return W
def _str2coords(s, names=('id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z')):
	"""
	Input:
		s: PredictionString (e.g. from train dataframe)
		names: array of what to extract from the string
	Output:
		list of dicts with keys from `names`
	"""
	coords = []
	for l in np.array(s.split()).reshape([-1, 7]):
		coords.append(dict(zip(names, l.astype('float'))))
		if 'id' in coords[-1]:
			coords[-1]['id'] = int(coords[-1]['id'])
	return coords


def load_anno_idx(idx, train, all_imgs_coords_points, draw=False, draw_dir='/home/ahkamal/Desktop/mask_samples/'):
	
    camera_matrix = np.array([[3701.25, 0, 1692.0], 
                            [0, 3701.25, 615.0],
                            [0, 0, 1]], dtype=np.float32)
    bottom_half = 0
    image_shape = (1230, 3384)
    img_prefix = '/home/ahkamal/Desktop/rendered_image/Cam.000/train/'
    img_format = '.png'

    out_dir = '/data/ahkamal/Kaggle_PKU_Baidu_v1/data/Kaggle/pku-autonomous-driving/'
    
    labels = []
    bboxes = []
    rles = []
    eular_angles = []
    quaternion_semispheres = []
    translations = []
    
    img_name = img_prefix + train['ImageId'].iloc[idx] + img_format
    
    car_model_dir = os.path.join(out_dir, 'car_models_json')
    car_model_dict = {}
    for car_name in os.listdir(car_model_dir):
        with open(os.path.join(out_dir, 'car_models_json', car_name)) as json_file:
            car_model_dict[car_name[:-5]] = json.load(json_file)

    if not os.path.isfile(img_name):
        assert "Image file does not exist!"
    else:
        if draw:
            image = imread(img_name)
            mask_all = np.zeros(image.shape)
            merged_image = image.copy()
            alpha = 0.8  # transparency

            gt = _str2coords(train['PredictionString'].iloc[idx])
        
        for gt_pred in gt:
            eular_angle = np.array([gt_pred['yaw'], gt_pred['pitch'], gt_pred['roll']])
            translation = np.array([gt_pred['x'], gt_pred['y'], gt_pred['z']])
            quaternion = euler_angles_to_quaternions(eular_angle)
            quaternion_semisphere = quaternion_upper_hemispher(quaternion)

            new_eular_angle = quaternion_to_euler_angle(quaternion_semisphere)
            distance = RotationDistance(new_eular_angle, eular_angle)
            # distance = np.sum(np.abs(new_eular_angle - eular_angle))
            if distance > 0.001:
                print("Wrong !!!", img_name)

            labels.append(gt_pred['id'])
            eular_angles.append(eular_angle)
            quaternion_semispheres.append(quaternion_semisphere)
            translations.append(translation)
            # rendering the car according to:
            # Augmented Reality | Kaggle

            # car_id2name is from:
            # https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/car_models.py
            car_name = car_id2name[gt_pred['id']].name
            vertices = np.array(car_model_dict[car_name]['vertices'])
            # print(len(vertices))
            vertices[:, 1] = -vertices[:, 1]		
            triangles = np.array(car_model_dict[car_name]['faces'])-1
                
            img_coords_points = all_imgs_coords_points[train['ImageId'].iloc[idx]]

            # project 3D points to 2d image plane
            yaw, pitch, roll = gt_pred['yaw'], gt_pred['pitch'], gt_pred['roll']
            # I think the pitch and yaw should be exchanged
            yaw, pitch, roll = -pitch, -yaw, -roll # ADDED - original --> -pitch, -yaw, -roll

            # Rt = np.eye(4)
            # t = np.array([gt_pred['x'], gt_pred['y'], gt_pred['z']])
            # Rt[:3, 3] = t
            # Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
            # Rt = Rt[:3, :]
            # P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
            # P[:, :-1] = vertices
            # P = P.T

            # img_coords_points = np.dot(camera_matrix, np.dot(Rt, P))
            # img_coords_points = img_coords_points.T
            # img_coords_points[:, 0] /= img_coords_points[:, 2]
            # img_coords_points[:, 1] /= img_coords_points[:, 2]
        
            # project 3D points to 2d image plane
            x1, y1, x2, y2 = img_coords_points[:, 0].min(), img_coords_points[:, 1].min(), img_coords_points[:,
                                                                                    0].max(), img_coords_points[:,
                                                                                            1].max()
            bboxes.append([x1, y1, x2, y2])

            # project 3D points to 2d image plane
            
            if draw:
                # project 3D points to 2d image plane
                mask_seg = np.zeros(image.shape, dtype=np.uint8)
                mask_seg_mesh = np.zeros(image.shape, dtype=np.uint8)
                for t in triangles:
                    coord = np.array([img_coords_points[t[0]][:2], img_coords_points[t[1]][:2], img_coords_points[t[2]][:2]],
                                    dtype=np.int32)
                    # This will draw the mask for segmenation
                    cv2.drawContours(mask_seg, np.int32([coord]), 0, (255, 255, 255), -1)
                    cv2.polylines(mask_seg_mesh, np.int32([coord]), 1, (0, 255, 0))

                mask_all += mask_seg_mesh

                ground_truth_binary_mask = np.zeros(mask_seg.shape, dtype=np.uint8)
                ground_truth_binary_mask[mask_seg == 255] = 1
                if bottom_half > 0:  # this indicate w
                    ground_truth_binary_mask = ground_truth_binary_mask[int(bottom_half):, :]

                # x1, x2, y1, y2 = mesh_point_to_bbox(ground_truth_binary_mask)

                # Taking a kernel for dilation and erosion,
                # the kernel size is set at 1/10th of the average width and heigh of the car

                kernel_size = int(((y2 - y1) / 2 + (x2 - x1) / 2) / 10)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                # Following is the code to find mask
                ground_truth_binary_mask_img = ground_truth_binary_mask.sum(axis=2).astype(np.uint8)
                ground_truth_binary_mask_img[ground_truth_binary_mask_img > 1] = 1
                ground_truth_binary_mask_img = cv2.dilate(ground_truth_binary_mask_img, kernel, iterations=1)
                ground_truth_binary_mask_img = cv2.erode(ground_truth_binary_mask_img, kernel, iterations=1)
                fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask_img)
                encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)

                encoded_ground_truth['counts'] = encoded_ground_truth['counts'].decode("utf-8")
                rles.append(encoded_ground_truth)
                
        
        if draw:
        # if False:
            mask_all = mask_all * 255 / mask_all.max()
            # mask_all = cv2.flip(mask_all,0)
            cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image)

            for box in bboxes:
                cv2.rectangle(merged_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                            thickness=5)

            imwrite(merged_image, os.path.join(draw_dir, train['ImageId'].iloc[idx] + '.jpg'))
            # imwrite(mask_all.astype(np.uint8),'/home/ahkamal/Desktop/test7.jpg')

        if len(bboxes):
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            eular_angles = np.array(eular_angles, dtype=np.float32)
            quaternion_semispheres = np.array(quaternion_semispheres, dtype=np.float32)
            translations = np.array(translations, dtype=np.float32)
            assert len(gt) == len(bboxes) == len(labels) == len(eular_angles) == len(quaternion_semispheres) == len(translations)
            
            annotation = OrderedDict([
                ('filename', img_name),
                ('width', image_shape[1]),
                ('height', image_shape[0]),
                ('bboxes', bboxes.tolist()),
                ('labels', labels.tolist()),
                ('eular_angles', eular_angles.tolist()),
                ('quaternion_semispheres', quaternion_semispheres.tolist()),
                ('translations', translations.tolist()),
                ('rles', [{'size':rles[0]['size'],'counts':str(rles[0]['counts'])}])
            ])
            return annotation
    

if __name__ == '__main__':
    all_annots = []
    train = pd.read_csv('/home/ahkamal/Desktop/rendered_image/Cam.000/_train.csv')
    all_imgs_coords_points = np.load('/home/ahkamal/Desktop/rendered_image/all_imgs_coords_points.npz')

    for i in tqdm(range(len(train))):
        annotation = load_anno_idx(i, train, all_imgs_coords_points, draw=True)
        all_annots.append(annotation)

    with open('/home/ahkamal/Desktop/_train.json','w') as file:
        json.dump(all_annots,file,indent=4)
        file.write('\n')
    file.close()
    print('Done')
