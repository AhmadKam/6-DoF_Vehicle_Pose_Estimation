"""
Generates annotations csv file based using json data (for single car prediction)
"""

import json
import os
import csv

# Validation dataset directory
source = '/home/ahkamal/Desktop/rendered_image/Cam.000/val/' 

# Annotation file for validation dataset (json)
with open('/home/ahkamal/Desktop/rendered_image/Cam.000/_val.json','r') as file:
	data = json.load(file)

# Store filenames of validation images
imgs = os.listdir("/home/ahkamal/Desktop/rendered_image/Cam.000/val/")

for i,f in enumerate(imgs):
	with open('/home/ahkamal/Desktop/rendered_image/Cam.000/_val.txt', 'a') as file:
			file.write(f)
			file.write('\n')
	file.close()

for i,d in enumerate(data):
	if i == 0:
		with open('/home/ahkamal/Desktop/rendered_image/Cam.000/_val.csv', 'a') as file:
			writer = csv.writer(file)
			writer.writerow(['ImageId','PredictionString'])
		file.close()
	for j in imgs:
		if os.path.join(source,j) == d['filename']:
			print(1)
			model_num = d['labels']
			euler_annot = d["eular_angles"][0]
			transl_annot = d['translations'][0]
			#Write to CSV
			with open('/home/ahkamal/Desktop/rendered_image/Cam.000/_val.csv', 'a') as file:
				writer = csv.writer(file)
				# Write values in non-scientific form
				euler_nonsci = [float('{:f}'.format(x)) for x in euler_annot]
				transl_nonsci = [float('{:f}'.format(x)) for x in transl_annot]
				m = model_num, euler_nonsci, transl_nonsci
				writer.writerow([j[:-4], str(m).replace(',','').replace('(','').replace(')','').replace('[','').replace(']','')])
			file.close()	
			break
		else:
			continue

print('Done')
	

