import os, sys
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import glob
import shutil

sys.path.append('/data/ahkamal/6-DoF_Vehicle_Pose_Estimation_Through_Deep_Learning')
from configs.htc.htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_wudi import work_dir

scene_setup = 'FC5' # e.g. FC5 | BL10 | FL10 | FC7 etc.

num_epochs = 19

# Get current date and time
time = '{}'.format(datetime.now().strftime("%b%d-%H-%M"))

files_list = glob.glob('{}*'.format(work_dir))
current_dir = max(files_list,key=os.path.getmtime)
usr_dir = os.path.dirname(os.path.dirname(current_dir)) # user directory

# Provide directory to logs
epoch_loss_source = '{}/epoch_losses.log'.format(current_dir)
rot_loss_source = '{}/rot_losses.log'.format(current_dir)
transl_loss_source = '{}/transl_losses.log'.format(current_dir)

map_source = '{}/mAP.log'.format(current_dir)

destination = '{}/tf_log/{}_v{}/'.format(usr_dir,time,scene_setup)

if not os.path.exists(destination):
    os.makedirs(destination)

shutil.copy(epoch_loss_source, destination)
shutil.copy(rot_loss_source, destination)
shutil.copy(transl_loss_source, destination)

shutil.copy(map_source, destination)

writer = SummaryWriter(log_dir=destination)


epoch_losses = []
with open(destination+'epoch_losses.log','r') as file:
    epoch_losses = file.readlines()
epoch_losses = [float(x.rstrip()) for x in epoch_losses]

rot_losses = []
with open(destination+'rot_losses.log','r') as file:
    rot_losses = file.readlines()
rot_losses = [float(x.rstrip()) for x in rot_losses]

transl_losses = []
with open(destination+'transl_losses.log','r') as file:
    transl_losses = file.readlines()
transl_losses = [float(x.rstrip()) for x in transl_losses]

maps = []
with open(destination+'mAP.log','r') as file:
    maps = file.readlines()
maps = [float(x.rstrip()) for x in maps]


for n_iter in range(num_epochs):
    writer.add_scalar('Training Loss/epoch', epoch_losses[n_iter], n_iter+1)
    writer.add_scalar('Training Loss/rot', rot_losses[n_iter], n_iter+1)
    writer.add_scalar('Training Loss/transl', transl_losses[n_iter], n_iter+1)
    writer.add_scalar('mAP', maps[n_iter], n_iter+1)
