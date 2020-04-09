import os
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import shutil

batch_size = 1
num_epochs = 43
num_imgs = 800
num_iter = 800 # loss intervals for plotting


time = '{}'.format(datetime.now().strftime("%b%d-%H-%M"))
source = '/home/ahkamal/Desktop/losses_log.txt'
destination = '/data/ahkamal/tf_log/{}/'.format(time)

if not os.path.exists(destination):
	os.makedirs(destination)

shutil.move(source, destination)

writer = SummaryWriter(log_dir=destination)
  

loss = []

with open(destination+'losses_log.txt','r') as file:
	loss = file.readlines()

checkpoint = 0
l = []
	
loss = [float(x.rstrip()) for x in loss]

for i in range(len(loss)):

	if i % num_iter == 0 and i!=0 :
		l.append(sum(loss[checkpoint:i])/(i-checkpoint))
		checkpoint = i
	else:
		continue

# x axis values 
epoch = np.arange(0, num_epochs, num_epochs/len(l)) # number of successful epochs

for n_iter in range(len(epoch)):
    writer.add_scalar('Loss/train', l[n_iter], n_iter)

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




