import os
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import shutil

batch_size = 1
num_epochs = 12
num_imgs = 1600
num_iter = 1600 # loss intervals for plotting


time = '{}'.format(datetime.now().strftime("%b%d-%H-%M"))
loss_source = '/home/ahkamal/Desktop/losses_log.txt'
map_source = '/home/ahkamal/Desktop/mAP_log.txt'
destination = '/data/ahkamal/tf_log/{}/'.format(time)

if not os.path.exists(destination):
	os.makedirs(destination)

shutil.move(loss_source, destination)
shutil.move(map_source, destination)

writer = SummaryWriter(log_dir=destination)
  

loss = []
mAP = []

with open(destination+'losses_log.txt','r') as file:
	loss = file.readlines()

with open(destination+'mAP_log.txt','r') as file:
	mAP = file.readlines()

checkpoint = 0
l = []
	
loss = [float(x.rstrip()) for x in loss]
mAP = [float(x.rstrip()) for x in mAP]

for i in range(len(loss)):

	if (i+1) % num_iter == 0 and i!= 0 :
		l.append(sum(loss[checkpoint:i])/(i-checkpoint))
		checkpoint = i
	else:
		continue

# x axis values 
epoch = np.arange(0, num_epochs, num_epochs/len(l)) # number of successful epochs

for n_iter in range(len(epoch)):
    writer.add_scalar('Loss/train', l[n_iter], n_iter)
    writer.add_scalar('mAP/val',mAP[n_iter],n_iter+1)

"""
Matplotlib plot

# plotting the points 
plt.plot(epoch, l) 
  
# naming the x axis 
plt.xlabel('Epochs') 
# naming the y axis 
plt.ylabel('Training Loss') 
  
# function to show the plot 
plt.show() 
"""




