"""
Ethan Pullen & Dhruv Joshi

CS465 Project 1

This file makes the watermark detecting neural network
"""
import cv2
import random
import numpy as np
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras import backend as K
from keras.optimizers import Adam
import sys
import os
import skimage.measure
import keras.backend as K


# train_datagen  = ImageDataGenerator()
# test_datagen = ImageDataGenerator()
    
# train_generator = train_datagen.flow_from_directory(
#         '/content/train/',
#         target_size=(230, 230),#The target_size is the size of your input images,every image will be resized to this size
#         batch_size=32,
#         class_mode='categorical')
# validation_generator = test_datagen.flow_from_directory(
#         '/content/validation/',
#         target_size=(230, 230),#The target_size is the size of your input images,every image will be resized to this size
#         batch_size=32,
#         class_mode='categorical')
# print(train_generator)
# model2 = Sequential()
# model2.add(Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) 
# model2.add(Conv2D(8, (3, 3), activation='relu'))
# model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(Dropout(0.25))
# #--------------------------
# model2.add(Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) 
# model2.add(Conv2D(8, (3, 3), activation='relu'))
# model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(Dropout(0.25))
# #--------------------------
# model2.add(Flatten())
# model2.add(Dense(16, activation='relu'))
# model2.add(Dropout(0.5))
# model2.add(Dense(2, activation='softmax'))
# model2.summary()#prints the summary of the model that was created
# model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# model2.fit_generator(
#         train_generator,
#         steps_per_epoch=2000,
#         epochs=65, validation_data=validation_generator
#         )

batch_size = 64
img_rows, img_cols = 230, 230
epochs = 12

def main(args):

	# load the data
	train_real, train_wmi, train_wmm = load_files(args[1])
	test_real, test_wmi, test_wmm = load_files(args[2])
	train_wmi = np.array(train_wmi)
	test_wmi = np.array(test_wmi)
	train_wmm = np.array(train_wmm)
	test_wmm = np.array(test_wmm)


	# make the model
	wmd = Sequential()
	wmd.add(Conv2D(128, kernel_size = (7, 7), activation='relu', input_shape = (img_rows, img_cols, 3)))
	wmd.add(MaxPooling2D(pool_size=(8, 8)))
	wmd.add(Conv2D(4, kernel_size = (3, 3), activation='relu'))
	wmd.add(Conv2D(1, kernel_size = (3, 3), activation='relu'))
	wmd.add(Reshape((24, 24)))
	wmd.compile(loss = custom_loss_function(1.5), optimizer=Adam(), metrics=['accuracy']) 
	wmd.fit(train_wmi, train_wmm, batch_size = batch_size, epochs = epochs, verbose = 1, validation_split = 0.1)

	# evaluate training and testing loss and accuracies
	score = wmd.evaluate(train_wmi, train_wmm, verbose = 0)
	print("Train Loss: ", score[0])
	print("Train Accuracy: ",score[1])

	score = wmd.evaluate(test_wmi, test_wmm, verbose = 0)
	print("Test Loss: ", score[0])
	print("Test Accuracy: ",score[1])

	# save the model
	name = "models/wmd_data"+str(train_wmi.shape[0])+"mult"+args[3]+".h5"
	wmd.save(name)

	return

def custom_loss_function(multiplier):
	def custom_loss(y_true, y_pred):
		return custom_loss_val(y_true, y_pred, multiplier)
	return custom_loss

def custom_loss_val(y_true, y_pred, multiplier):
	y_true = K.flatten(y_true)
	y_pred = K.flatten(y_pred)

	# count number of marked and unmarked pixels
	mp = K.sum(y_true)
	nmp = 576 - mp

	diff = y_true - y_pred
	false_negative = (K.relu(diff))
	false_positive = (diff - false_negative)

	fne = K.sum(K.square(false_negative))
	fpe = K.sum(K.square(false_positive))

	return (fne*float(multiplier) + fpe)


def load_files(path):
	''' 
	This function takes in a path and returns lists of the images, watermarked images, and watermark masks 
	for all images in the file
	'''
	files = os.listdir(path)
	img_list = []
	wmimg_list = []
	wmm_list = []
	for i,f in enumerate(files):
		img, wmimg, wmm = watermark(cv2.resize(load_img(path + "/" + f), (230,230)))
		img_list.append(img)
		wmimg_list.append(wmimg)
		wmm_list.append(wmm)
		if i % 1000 == 0:
			print(f)
			print(i)
	
	return img_list, wmimg_list, wmm_list

def load_img(url):
	''' 
	This function takes in a url of an image and returns the image 
	'''
	return cv2.imread(url)

def watermark(img):
	'''
	This function takes in an image and returns the image, the image with a watermark,
	and just the watermark mask. 

	The water mark is made to vary the font, fontface, color, opacity, 
	text_size, text_length, text, thickness, and it's location
	'''
	if img is None:
		return
	wmm = np.zeros((230,230,3), np.uint8)
	fontFace = int(5 * random.random())
	color = (int(255*random.random()),int(255*random.random()), int(255*random.random()))
	opacity = .8 + .2*random.random()
	text_size = .25 + random.random()
	text_length = 6+6*random.random()
	text = ""
	x = int(10+random.random()*200)
	y = int(10+random.random()*200)
	thickness = int(1 + 4*random.random())
	for i in range(int(text_length)):
		text = text + chr(int(13 + 243*random.random()))

	wmimg = img.copy()
	a = img.copy()
	cv2.putText(wmm,text, (x,y), fontFace, text_size, color, thickness = thickness)
	cv2.putText(a,text, (x,y), fontFace, text_size, color, thickness = thickness)
	cv2.addWeighted(wmimg, 1-opacity, a, opacity,
		0, wmimg)
	wmm = cv2.resize(wmm,(24,24))
	wmm = cv2.cvtColor(wmm, cv2.COLOR_BGR2GRAY)
	# wmm = cv2.resize(wmm, (24,24))
	ret, wmm = cv2.threshold(wmm, 1,255, cv2.THRESH_BINARY)

	# add back the 3rd channel
	#wmm = wmm[:, :, np.newaxis]

	return img, wmimg, wmm

def show(img):
	'''
	This function displays the image passed to it
	'''
	cv2.imshow("image", img)
	cv2.waitKey(0)


if __name__ == "__main__":
	main(sys.argv)



