# Implementation
**GENERAL NOTE**: When referring to the configuration file, it corresponds to the following:
```
../configs/htc/htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_wudi.py
```
## Folder Structure
* Blender Script<br />
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

* Dataset Images (this directory will be generated automatically by the Blender script)
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
## Dataset Creation in Blender
**NOTE**: To access the Python API in Blender, change from "Default" to "Scripting" using the "Screen layout" menu in the top menu bar (next to "File", "Render", etc.)
1. Open [blender/van_model_render.blend](../blender/van_model_render.blend)
2. Vehicle model setup:
    1. Import the vehicle CAD model into Blender twice. One model will remain as is, and the other model will be decimated (lower face count) to obtain the vehicle’s 2D mask.
    2. For the unchanged model:
        1. Select the model and set its origin by selecting “Origin to Geometry” from the “Tools” tab on the far left side of the screen.
        2. Set the cursor to the model’s origin by selecting “Cursor to Selected” from the “Snap” menu from “Object” in the bottom menu bar.
        3. Create an “Empty” object from the “Create” tab on the far left side of the screen. Ensure that the “Empty” object and vehicle model origins overlap.
        4. In the “Scene” tree on the far right side of the screen, select the “Empty” object, **THEN** the vehicle object.
        5. Select “Parent to Object” from the “Parent” menu from “Object” in the bottom menu bar. 
        6. In the “Scene” tree, select the camera object being used to render images (e.g. Cam.000)
        7. Select the “Constraints” tab (*link* icon) from the section underneath the “Scene” tree. Once done, as the vehicle’s position changes, the “Empty” object should adjust its position accordingly.
        8. Select “Track To” from the “Add Object Constraint” menu.
        9. Set the “Target” to the “Empty” object created in Step iii. Also, ensure that “To” is to “-Z”, and “Up” to “Y”. Once done, as the vehicle’s position changes, the camera should adjust accordingly.
    3. For the decimated model:
        1. Repeat step 2.ii.a) above.
        2. Select the “Modifiers” tab (*spanner* icon) on the far right of the screen, underneath the “Scene” tree.
        3. Select “Decimate” from the “Add Modifier” menu.
        4. Set the ratio to reduce the number of faces of the model (van model was reduced to 2233 aces from 535801 faces in this capstone).
    4. In the “Scene” tree, make sure to select the *eye* icon to show the model, and select the *render* icon (camera) for the unchanged model. And for the decimated model, make sure to hide it and deselect the render icon.
    5. Check that both vehicle models are set to the same pose in the world frame.
3. For `render_imgs()`:
    1. Set the function parameter in `main()` to either `single-scale` or `multi-scale` based on camera setup requirements for the dataset.
    2. Set image resolution.
    3. Set the active camera variable `activeCamera` used to render the images.
    4. Set the dataset directory `ds_dir`. This is where the dataset will be located the cluster node.<br />
    **NOTE**: This must be the same as the `ds_dir` variable set in the configuration file.
    5. Set the root directory `root_dir` for saving the dataset.
    6. Set the vehicle type number `model_num` based on car_model.py.
    7. Set the `thresh` variable to total number of required images.
    8. Set the `frames` variable based on required camera position (e.g. 360°, front centre, back left, etc.).
    9. Set constraints:
        1. For multi-scale:
            Set second parameter of `new_scale` variable to maximum camera distance.
        2. For single-scale:
            Set the `new_scale` variable to the desired camera distance from the van.
    10. Click on the "Run Script" button. The camera images will now render. This may take some time based on the total number of images required. If you open the terminal, a progress bar showing the progress of the rendering process along with an ETA is provided.
    11. Once all images are rendered, the images are split (80-10-10) into train, validation, and test datasets respectively. Go to the directory assigned in the script to find the dataset images, as well as their groundtruth annotation files.
    12. Check that the image directories defined in the *.json* annotation files (e.g _train.json) are the same as `ds_dir` defined in the configuration file.
4. For `get_model_data()`:
    1. Set the `vehicle` variable to the desired object to be rendered. The name can be obtained from the “Scene” tree on the far right of the screen (make sure to use the non-decimated model).
    2. Check that the desired object is not hidden.
    3. Set the `model_name` variable to the vehicle name (e.g. Mercedes Vito).
    4. Set the `model_type` variable to the vehicle type (e.g. van, SUV, sedan).
    5. Click on the "Run Script" button. The script outputs a *.json* file.
5. For `get_calibration_matrix_K()`:
    1. Set `currentCameraObj` variable to the camera for which the calibration parameters are needed.
    2. Click on the "Run Script" button. The camera parameters will be generated in the terminal.

### Image Naming Convention
1. Multi-Scale Dataset - For image `van_Zx_sy_fz.png`:
    * *x* is the circle height.
    * *y* is the circle scale.
    * *z* is the frame in which the image is rendered.
    
2. Single-Scale Dataset - For image `van_Xx_Yy_Rz.png`:
    * *x* is the translation along the X-axis.
    * *y* is the translation along the Y-axis.
    * *z* is the translation about the Z-axis.

## Directories
The directories in the files below need to be set:

1. In the configuration file in [../configs/htc/](../configs/htc/)
* Set the path to the dataset directory `ds_dir`.
* Set the path to load the HRNetV2p model in `load_from`. The model can be found [here](https://drive.google.com/file/d/17qN0pB9Tp0DFBonEgDWx_JRFEfP8wT_E/view?usp=sharing) and needs to be added to [../configs/htc/](../configs/htc/)
* IF NEEDED: Set the path to resume the epoch in `resume_from`.

**NOTE**: 
- `resume_from` loads both the model weights and optimizer status at the epoch of the respective checkpoint. It is useful for resuming the training process if it is accidentally interrupted.
- `load_from` only loads the model weights and the training epoch starts from 0; it is used to load the pre-trained HRNet model. It can also be used for fine-tuning the network.

2. In `kaggle_pku.py`:
    The file can be found in `../mmdet/datasets/` in the Anaconda env.
    * Set the `out_dir` path in `load_annotations()`.

## Adding a New Vehicle Model for Training/Testing
1. Given the vehicle’s CAD model, get the model data *.json* file as per Step 4 in Dataset Creation above.
2. Add the new vehicle label to `models` in `car_models.py` in [/blender](../blender/car_models.py).
3. In `kaggle_pku.py`:
    The file can be found in `../mmdet/datasets/` in the Anaconda env.
    1. Add the vehicle ID to `unique_car_mode` in `load_annotations()`.<br />
**NOTE**: The vehicle ID assigned in Blender **must** be used for Steps 2 and 3.

## Training
1. In the configuration file:
    1. Set the translation and rotation weights `translation_weight` and `rot_weight`, respectively.
    2. Set image resize resolution in `train_pipeline`.
    3. Set the dataset directory `ds dir`.
    4. Set the batchsize `imgs_per_gpu` in `data`
    5. Set the learning rate `lr` in `optimizer`.
    6. Set the learning decay steps `step` in `lr_config`
    7. Set the logging interval `interval` in `log_config`.
    8. Set the number of epochs for training `total_epochs`.
    9. Set work directory `work_dir`.
    10. Load the pre-trained HRNet model using `load_from`.
    The model can be found in `../configs/htc/`.
    11. IF NEEDED, set the resuming checkpoint path using `resume from`.
2. In `kaggle_pku.py`:
    The file can be found in `../mmdet/datasets/` in the Anaconda env.
    1. Set the path to the `car_models.py` (located in [/blender](../blender/car_models.py))
    2. Set the original image resolution (as per the Blender script) in `image_shape`.
3. For validation:
    1. Set the bounding box confidence threshold `conf_thresh` in `evaluation` in the configuration file.
    2. Set repository folder path to import scripts in `kaggle_hooks.py`.
    The file can be found in `../mmdet/core/evaluation/` in the Anaconda env.
    3. Set the translation and rotation thresholds `thres_tr_list` and `thres_ro_list`, respectively in `map_calculation.py`.
    The file can be found in `../mmdet/utils/` in the Anaconda env.
    

### Loss Description
Below is a description of the losses recorded during training:
* `loss_rpn_cls` is the HRNet loss calculated when predicting the object class (e.g. car, bike, plane, etc.).
* `loss_rpn_bbox` is the HTC loss calculated when predicting the object's bounding box.
* `sx.loss_cls` is the HTC loss calculated when predicting the object class (e.g. car, bike, plane, etc.) at stage *x*.
* `sx.acc` is the network accuracy in correctly predicting the object class (out of 100) at stage *x*.
* `sx.loss_bbox` is the HTC loss calculated when predicting the object's bounding box at stage *x*.
* `sx.loss_mask` is the HTC loss calculated when predicting the object's mask at stage *x*.

* `sx.loss_quaternion` is the rotation head loss calculated when predicting the object's quaternion rotation at stage *x*.
* `sx.rotation_distance` is the distance between the ground truth and relative rotations at stage *x*.
* `sx.loss_translation` is the translation head loss calculated when predicting the object's translation at stage *x*.
* `sx.translation_distance` is the absolute distance between the ground truth and relative translations at stage *x*.
* `sx.translation_distance_relative` is the relative distance between the ground truth and relative translations at stage *x*.

## Testing
1. In the configuration file:
    1. Set the image resolution in `img_scale` in `test_pipline`.
    2. Check that the dataset directory `ds_dir` is set in the configuration file.
2. In `map_calculation.py`:
    The file can be found in `../mmdet/utils/` in the Anaconda env.
    1. Set the translation and rotation thresholds `thres_tr_list` and `thres_ro_list`, respectively.
3. 2. In `test_kaggle_pku.py`:
    1. Set the checkpoint path `checkpoint_path` in `parse_args()`.
    2. Set the bounding box confidence threshold `conf_thresh` when calling `write_submission`.
    3. Optional arguments:
        1. Set `plot=True` in `main()` to visualise all predictions.
        2. Set `save_img=True` in `main()` to save all predictions.
        3. Set `plot_tp=True` when calling `write_submission` to visualise **only** true positive predictions.
        4. Set `save_tp=True` when calling `write_submission` to save **only** true positive predictions.
