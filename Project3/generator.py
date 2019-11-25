'''
Ethan Pullen & Dhruv Joshi
generator.py
'''

import numpy as np
from skimage.io import imread
import json
from keras.utils import Sequence

class Data_Gen(Sequence):

    def __init__(self, path, r, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.r = r

    def __len__(self):
        return np.ceil(r[1]-r[0] / float(self.batch_size))

    def __getitem__(self, idx):
        r = self.r[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for i in range(r[0],r[1]):
        	with open(self.path+"/Descriptions/ISIC_"+'{:07d}'.format(10)) as file:
        		temp = json.load(file)
        		batch_x.append([float(temp['meta']['clinical']['age_approx'])/100, 1 if temp['meta']['clinical']['sex'] == 'm' else 0])
        		batch_y.append(1 if temp['meta']['clinical']['benign_malignant'] == 'malignant' else 0)

        return [np.array(batch_x)
        	, np.array([
            imread(self.path+"/Images/ISIC_"+'{:07d}'.format(10)+".jpeg")
               for file_name in batch_x])], np.array(batch_y)