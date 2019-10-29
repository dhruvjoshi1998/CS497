"""
Authors: Dhruv Joshi & Ethan Pullen
Date: 29 October 2019
File: gen.py
Purpose: Generate words from the saved models
"""

import sys
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from scipy.io.wavfile import write

NOISE_DIM = 20

def load(word, epoch):
	return load_model("models/"+word+"_gen_"+str(epoch)+".h5")

def generate_word(word, epoch, count):
	generator = load(word, epoch)
	gen_noise = np.random.normal(0, 1, (count, NOISE_DIM))
	synthetic_data = generator.predict(gen_noise)

	for i in range(len(synthetic_data)):
		save_word(synthetic_data[i], word+"_"+str(i))

def save_word(data_point, path):
	write("output/"+path+".wav", 16000, data_point)

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print("ERR: Usage -- python3 gen.py <word> <model_epoch> <number_of_words>")	
		exit(-1)

	generate_word(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))


