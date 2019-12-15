'''
Ethan Pullen & Dhruv Joshi
generator.py
'''

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import json
from keras.utils import Sequence
import os
import random

class Data_Gen(Sequence):

	def __init__(self, path, r, batch_size):
		self.indices = [x for x in range(r[0], r[1])]
		self.classes = np.empty((r[1]-r[0],))
		random.shuffle(self.indices)
		# print(self.indices)
		self.path = path
		self.batch_size = batch_size
		self.r = r

	def __len__(self):
		return (self.r[1]-self.r[0]) // self.batch_size

	def __getitem__(self, idx):
		# r = (idx * self.batch_size + self.r[0],(idx + 1) * self.batch_size + self.r[0])
		# print(idx*self.batch_size, (idx+1)*self.batch_size)
		batch_x = []
		batch_x1 = []
		batch_y = []
		total_mal = 0
		for i in [self.indices[x] for x in range(idx*self.batch_size, (idx+1)*self.batch_size)]:
			# print(i)
			try:
				with open(self.path+"/Descriptions/ISIC_"+'{:07d}'.format(i)) as file:
					temp = json.load(file)
				
					total_mal += 1 if temp['meta']['clinical']['benign_malignant'] == "malignant" else 0
					x = temp['meta']['clinical']['age_approx']
				batch_x1.append(imread(self.path+"/Images/ISIC_"+'{:07d}'.format(i)+".jpeg"))
				if temp['meta']['clinical']['age_approx'] == None:
					batch_x.append(np.array([0, 1 if temp['meta']['clinical']['sex'] == 'm' else 0]))
				else:
					batch_x.append(np.array([float(temp['meta']['clinical']['age_approx'])/100, 1 if temp['meta']['clinical']['sex'] == 'm' else 0]))
				

				batch_y.append(np.array([1]) if temp['meta']['clinical']['benign_malignant'] == 'malignant' else np.array([0]))

			except FileNotFoundError:
					print("Missing",i)
					continue
			except KeyError:
					continue
			# if batch_x1[-1].shape[0] != 768 or batch_x1[-1].shape[1] != 1024:
			# 	print(batch_x1[-1].shape)

		print("\nMal", total_mal)
		return [np.array(batch_x)
			, np.array(batch_x1)], np.array(batch_y)


def custom_generator(path, r, batch_size):
	i = 0
	values = range(r[0],r[1])
	while True:
		batch = {'images': [], 'features': [], 'labels': []}
		for b in range(batch_size):
			if i == len(values):
				i = 0
				random.shuffle(values)
			
			with open(self.path+"/Descriptions/ISIC_"+'{:07d}'.format(i)) as file:
				temp = json.load(file)
				if temp['meta']['clinical']['age_approx'] == None:
					continue
				batch['features'].append(np.array([float(temp['meta']['clinical']['age_approx'])/100, 1 if temp['meta']['clinical']['sex'] == 'm' else 0]))
				print("mal",temp['meta']['clinical']['benign_malignant'])
				batch['labels'].append(np.array([1]) if temp['meta']['clinical']['benign_malignant'] == 'malignant' else np.array([0]))
			batch['images'].append(resize(imread(self.path+"/Images/ISIC_"+'{:07d}'.format(i)+".jpeg"),(768,1024)))
			i += 1

		batch['images'] = np.array(batch['images'])
		batch['features'] = np.array(batch['features'])
		# Convert labels to categorical values
		batch['labels'] = np.array([batch['labels']])
		yield [batch['images'], batch['features']], batch['labels']

def resize_images(path, r):
	print("resizing:",r)
	cnt = 0
	for i in range(r[0],r[1]):
		if i%100 == 0:
			print (i)
		try:
			x = imread(path+"/ISIC_"+'{:07d}'.format(i)+".jpeg")
			imsave(path+"/ISIC_"+'{:07d}'.format(cnt)+".jpeg", resize(x, (384, 512)))
			cnt += 1
		except IOError:
			print("Missing",i)

def transfer_images(path, r):
	print("resizing:",r)
	cnt = 0
	for i in range(r[0],r[1]):
		if i%100 == 0:
			print (i)
		try:
			print(path+"/ISIC_"+'{:07d}'.format(i)+".jpeg")
			x = imread(path+"/ISIC_"+'{:07d}'.format(i)+".jpeg")
			imsave("P_data/Images/ISIC_"+'{:07d}'.format(cnt)+".jpeg", x)
			cnt += 1
		except IOError:
			print("Missing",i)
		

def transfer_info(path, r):
	print("transfering descriptions:",r)
	cnt = 0
	for i in range(r[0],r[1]):
		if i%100 == 0:
			print (i)
		try:
			with open(path+"/ISIC_"+'{:07d}'.format(i)) as r_file:
				x = json.load(r_file)
				with open("P_data/Descriptions/ISIC_"+'{:07d}'.format(cnt), 'w') as w_file:
					json.dump(x, w_file)

			cnt += 1
		except IOError:
			print("Missing",i)