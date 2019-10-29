"""
Authors: Dhruv Joshi & Ethan Pullen
Date: 29 October 2019
File: train.py
Purpose: Train and save models to Models folder
"""
import librosa
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.optimizers import Adam

import ganModels

# preprocess global values
DURATION = 4
SAMPLE_RATE = 16000
AUDIO_SHAPE = SAMPLE_RATE*DURATION

# train global values
EPOCHS = 21
BATCH_SIZE = 512

NOISE_DIM = 20

SAVE_INTERVAL = 3

def load_dataset(phrase):
	wav_paths = os.listdir("data/augmented_dataset/augmented_dataset/"+phrase)
	X = np.empty((len(wav_paths), AUDIO_SHAPE))
	for i, p in enumerate(wav_paths):
		if i % 100 == 0:
			print("Loading data:", i, "/", len(wav_paths))

		data, _ = librosa.core.load("data/augmented_dataset/augmented_dataset/"+phrase+"/"+p, sr=SAMPLE_RATE, res_type='kaiser_fast')

		# skip data longer than DURATION
		if len(data) > AUDIO_SHAPE:
			continue

		# make data longer if shorter than DURATION
		if AUDIO_SHAPE > len(data):
			max_offset = AUDIO_SHAPE - len(data)
			offset = np.random.randint(max_offset)
		else:
			offset = 0

		data = np.pad(data, (offset, AUDIO_SHAPE - len(data) - offset), "constant")
		X[i, ] = data
	print("Dataset Loaded -- Size: ", len(wav_paths))
	return X

def normalize_data(X):
	mean = X.mean()
	std = X.std()
	X = (X - mean)/std
	return X

def rescale(X):
	maximum = X.max()
	minimum = X.min()
	X = np.interp(X, (minimum, maximum), (-1, 1))
	return X

def train(data, phrase):

	# stats
	d_loss_mean = []
	g_loss_mean = []

	# make the models
	gen = ganModels.generator(NOISE_DIM, AUDIO_SHAPE)
	disc = ganModels.discriminator(AUDIO_SHAPE)
	gan_model = ganModels.stacked_G_D(gen, disc)

	gen.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.9), metrics=['accuracy'])
	disc.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.9), metrics=['accuracy'])
	gan_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.9), metrics=['accuracy'])

	for i in range(EPOCHS):
		print("starting epoch:", i, "/", EPOCHS)

		half_batch = BATCH_SIZE//2

		# Selecting random half batch location
		randi = np.random.randint(0, len(data) - half_batch)

		# Make noise for generator
		gen_noise = np.random.normal(0, 1, (half_batch, NOISE_DIM))

		real_audios = data[randi : randi+half_batch]
		synthetic_audios = gen.predict(gen_noise)

		xc_batch = np.concatenate((real_audios, synthetic_audios))
		yc_batch = np.concatenate((np.ones((half_batch, 1)), np.zeros((half_batch, 1))))

		print("prediscriminator train")

		# Train discriminator
		d_loss = disc.train_on_batch(xc_batch, yc_batch)
		gan_model.layers[1].set_weights(disc.get_weights())

		d_loss_mean.append(np.mean(d_loss))

		print("discriminator loss:", np.mean(d_loss))
		print("pregenerator train")


		# Train generator
		gen_noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))

		y_mislabeled = np.ones((BATCH_SIZE, 1))

		g_loss = gan_model.train_on_batch(gen_noise, y_mislabeled)
		gen.set_weights(gan_model.layers[0].get_weights())

		g_loss_mean.append(np.mean(g_loss))

		print("discriminator loss:", np.mean(d_loss))


		# Save model occasionally
		if i+1 % SAVE_INTERVAL == 0:

			print("Saving model at Epoch:", i)
			print("Discriminator loss:", np.mean(d_loss))
			print("Generator loss:", np.mean(g_loss))

			gen.save("models/"+phrase+"_gen_"+str(i)+".h5")
			disc.save("models/"+phrase+"_disc_"+str(i)+".h5")

	

if __name__ == "__main__":

	# preproccess data
	data = load_dataset("house")
	norm_data = normalize_data(data)
	rescaled_data = rescale(data)

	# train
	train(rescaled_data, "house")





