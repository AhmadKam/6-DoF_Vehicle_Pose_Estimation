"""
Generates json file holding entire dataset_name annotations
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import math
from math import sin, cos, acos, pi
from mmcv import imread, imwrite
from car_models import car_id2name
from pycocotools import mask as maskUtils
from scipy.spatial.transform import Rotation as R

def RotationDistance(p, g):

    if isinstance(p, np.ndarray) or isinstance(p, list):
        q1 = R.from_euler('xyz', p)
        q2 = R.from_euler('xyz', g)

    else:
        true = [g[0],g[1],g[2]]
        pred = [p[0],p[1],p[2]]
        q1 = R.from_euler('xyz', true)
        q2 = R.from_euler('xyz', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)

    W = (acos(W) * 360) / pi
    if W > 180:
        W = 360 - W
    return W

def euler_angles_to_quaternions(angle):
    """
    Convert euler angels to quaternions representation.
    yaw, pitch, roll
    
    Input:
        angle: n x 3 matrix, each row is [yaw, pitch, roll]
    Output:
        q: n x 4 matrix, each row is corresponding quaternion.
    """

    in_dim = np.ndim(angle)
    if in_dim == 1:
        angle = angle[None, :]

    n = angle.shape[0]

    yaw, pitch, roll = angle[:, 0], angle[:, 1], angle[:, 2]

    q = np.zeros((n, 4))

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q[:, 0] = cy * cr * cp + sy * sr * sp
    q[:, 1] = cy * sr * cp - sy * cr * sp
    q[:, 2] = cy * cr * sp + sy * sr * cp
    q[:, 3] = sy * cr * cp - cy * sr * sp

    if in_dim == 1:
        return q[0]
    return q

def quaternion_upper_hemispher(q):
    """
    The quaternion q and −q represent the same rotation be-
    cause a rotation of θ in the direction v is equivalent to a
    rotation of 2π − θ in the direction −v. One way to force
    uniqueness of rotations is to require staying in the “upper
    half” of S 3 . For example, require that a ≥ 0, as long as
    the boundary case of a = 0 is handled properly because of
    antipodal points at the equator of S 3 . If a = 0, then require
    that b ≥ 0. However, if a = b = 0, then require that c ≥ 0
    because points such as (0,0,−1,0) and (0,0,1,0) are the
    same rotation. Finally, if a = b = c = 0, then only d = 1 is
    allowed.
    :param q:
    :return:
    """
    a, b, c, d = q
    if a < 0:
        q = -q
    if a == 0:
        if b < 0:
            q = -q
        if b == 0:
            if c < 0:
                q = -q
            if c == 0:
                print(q)
                q[3] = 1

    return q

def quaternion_to_euler_angle(q):
    """
    Convert quaternion to euler angel.
    yaw, pitch, roll

    Input:
        q: 1 * 4 vector,
    Output:
        angle: 1 x 3 vector, each row is [yaw, pitch, roll]
    """
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    # return pitch, yaw, roll
    return yaw, pitch, roll

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


def load_anno_idx(root_dir, dataset_name, idx, ds_csv, all_imgs_coords_points, draw=False, draw_dir=None):
	
    draw_dir='{}/mask_samples/'.format(root_dir)

    camera_matrix = np.array([[3701.25, 0, 1692.0], 
                            [0, 3701.25, 615.0],
                            [0, 0, 1]], dtype=np.float32)
    bottom_half = 0
    image_shape = (1230, 3384)
    img_prefix = '{}/rendered_image/Cam.000/{}/'.format(root_dir,dataset_name)
    img_format = '.png'


    labels = []
    bboxes = []
    rles = []
    eular_angles = []
    quaternion_semispheres = []
    translations = []
    
    img_name = img_prefix + ds_csv['ImageId'].iloc[idx] + img_format
    
    car_model_dir = os.path.join(root_dir, 'car_models_json')
    car_model_dict = {}
    for car_name in os.listdir(car_model_dir):
        with open(os.path.join(root_dir, 'car_models_json', car_name)) as json_file:
            car_model_dict[car_name[:-5]] = json.load(json_file)

    if not os.path.isfile(img_name):
        assert "Image file does not exist!"
    else:
        if draw:
            image = imread(img_name)
            mask_all = np.zeros(image.shape)
            merged_image = image.copy()
            alpha = 0.8  # transparency

        gt = _str2coords(ds_csv['PredictionString'].iloc[idx])
        
        for gt_pred in gt:
            eular_angle = np.array([gt_pred['yaw'], gt_pred['pitch'], gt_pred['roll']])
            translation = np.array([gt_pred['x'], gt_pred['y'], gt_pred['z']])
            quaternion = euler_angles_to_quaternions(eular_angle)
            quaternion_semisphere = quaternion_upper_hemispher(quaternion)

            new_eular_angle = quaternion_to_euler_angle(quaternion_semisphere)
            distance = RotationDistance(new_eular_angle,eular_angle)

            assert distance < 0.001, "Incorrect quat to Euler angle conversion." # checks for correct quaternion conversion

            labels.append(gt_pred['id'])
            eular_angles.append(eular_angle)
            quaternion_semispheres.append(quaternion_semisphere)
            translations.append(translation)
            
            car_name = car_id2name[gt_pred['id']].name
            vertices = np.array(car_model_dict[car_name]['vertices'])
            vertices[:, 1] = -vertices[:, 1]		
            triangles = np.array(car_model_dict[car_name]['faces'])-1
                
            img_coords_points = all_imgs_coords_points[ds_csv['ImageId'].iloc[idx]]

        
            # project 3D points to 2d image plane
            x1, y1, x2, y2 = img_coords_points[:, 0].min(), img_coords_points[:, 1].min(), img_coords_points[:,
                                                                                    0].max(), img_coords_points[:,
                                                                                            1].max()
            bboxes.append([x1, y1, x2, y2])

            
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

                
                kernel_size = int(((y2 - y1) / 2 + (x2 - x1) / 2) / 10)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)

                # Get mask
                ground_truth_binary_mask_img = ground_truth_binary_mask.sum(axis=2).astype(np.uint8)
                ground_truth_binary_mask_img[ground_truth_binary_mask_img > 1] = 1
                ground_truth_binary_mask_img = cv2.dilate(ground_truth_binary_mask_img, kernel, iterations=1)
                ground_truth_binary_mask_img = cv2.erode(ground_truth_binary_mask_img, kernel, iterations=1)
                fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask_img)
                encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)

                encoded_ground_truth['counts'] = encoded_ground_truth['counts'].decode("utf-8")
               	rles.append(encoded_ground_truth)
                
        
        if draw:
       
            mask_all = mask_all * 255 / mask_all.max()
            cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image)

            for box in bboxes:
                cv2.rectangle(merged_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                            thickness=5)

            # imwrite(merged_image, os.path.join(draw_dir, ds_csv['ImageId'].iloc[idx] + '.jpg'))

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
                ('rles', [OrderedDict([('size',rles[0]['size']),('counts',str(rles[0]['counts']))])])
            ])
            return annotation
    

def load_annot(root_dir):
    datasets = ['train','val','test']
    all_imgs_coords_points = np.load('{}/rendered_image/all_imgs_coords_points.npz'.format(root_dir))

    for i in datasets:
        all_annots = []
        ds_csv = pd.read_csv('{}/rendered_image/Cam.000/_{}.csv'.format(root_dir,i))
        print('\nWriting {} DS Annoations\n'.format(i))
        for j in tqdm(range(len(ds_csv))):
            annotation = load_anno_idx(root_dir, i, j, ds_csv, all_imgs_coords_points, draw=True)
            all_annots.append(annotation)

        with open('{}/rendered_image/Cam.000/_{}.json'.format(root_dir,i),'w') as file:
            json.dump(all_annots,file,indent=4)
            file.write('\n')
        file.close()
