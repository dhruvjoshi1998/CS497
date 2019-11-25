'''
Ethan Pullen & Dhruv Joshi
generator.py
'''

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import json
from keras.utils import Sequence

class Data_Gen(Sequence):

	def __init__(self, path, r, batch_size):
		self.path = path
		self.batch_size = batch_size
		self.r = r

	def __len__(self):
		return (self.r[1]-self.r[0]) // self.batch_size

	def __getitem__(self, idx):
		r = self.r[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_x = []
		batch_x1 = []
		batch_y = []
		for i in range(r[0],r[1]):
			with open(self.path+"/Descriptions/ISIC_"+'{:07d}'.format(i)) as file:
				temp = json.load(file)
				if temp['meta']['clinical']['age_approx'] == None:
					continue
				batch_x.append(np.array([float(temp['meta']['clinical']['age_approx'])/100, 1 if temp['meta']['clinical']['sex'] == 'm' else 0]))
				batch_y.append(np.array([1]) if temp['meta']['clinical']['benign_malignant'] == 'malignant' else np.array([0]))
			batch_x1.append(resize(imread(self.path+"/Images/ISIC_"+'{:07d}'.format(i)+".jpeg"),(768,1024)))
			if batch_x1[-1].shape[0] != 768 or batch_x1[-1].shape[1] != 1024:
				print(batch_x1[-1].shape)

		return [np.array(batch_x)
			, np.array(batch_x1)], np.array(batch_y)