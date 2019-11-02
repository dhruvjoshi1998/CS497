import keras
import numpy as np
import sys
from keras.models import load_model
import watermarkdetection
import keras.losses
import cv2


keras.losses.custom_loss = watermarkdetection.custom_loss_function(5)


def main(args):
	model = load_model("models/wmd_data1000mult1.5.h5")

	img = cv2.resize(load_img(args[1]), (230, 230))

	_, wmi, wmm  = watermarkdetection.watermark(img)
	
	wmi = process_img(wmi)

	twmm = model.predict(wmi)[0]

	cv2.imshow("real_image", img)
	cv2.imshow("predicted", cv2.resize(twmm, (240, 240)))
	cv2.imshow("ground_truth", cv2.resize(wmm, (240, 240)))
	cv2.imshow("watermarked image", wmi[0])
	cv2.waitKey(0)

	return


def test_our_data(args):
	model_names = ["models/wmd_data1000mult1.h5", "models/wmd_data1000mult1.5.h5", "models/wmd_data1000mult2.h5", "models/wmd_data1000mult3.h5", "models/wmd_data1000mult3.h5", "pr_models/wmd_data1000mult1.h5", "pr_models/wmd_data1000mult2.5.h5", "pr_models/wmd_data1000mult3.h5", "pr_models/wmd_data1000mult4.h5", "pr_models/wmd_data1000mult5.h5"]

	models = []

	for model_name in model_names:
		models.append(load_model(model_name))
	
	for img_name in args[1:]:
		img = cv2.resize(load_img(img_name), (230, 230))
	
		_, wmi, wmm  = watermarkdetection.watermark(img)
		wmi = process_img(wmi)

		for i, model in enumerate(models):
			cv2.imshow(model_names[i][:10]+model_names[i][-6:-3], cv2.resize(model.predict(np.array([img]))[0], (240, 240)))
		cv2.imshow("ground_truth", cv2.resize(wmm, (240, 240)))
		cv2.imshow("watermarked image", wmi[0])
		cv2.waitKey(0)


def test_real_world(args):
	model_names = ["models/wmd_data1000mult1.h5", "models/wmd_data1000mult1.5.h5", "models/wmd_data1000mult2.h5", "models/wmd_data1000mult3.h5", "models/wmd_data1000mult3.h5", "pr_models/wmd_data1000mult1.h5", "pr_models/wmd_data1000mult2.5.h5", "pr_models/wmd_data1000mult3.h5", "pr_models/wmd_data1000mult4.h5", "pr_models/wmd_data1000mult5.h5"]

	models = []

	for model_name in model_names:
		models.append(load_model(model_name))
	
	for img_name in args[1:]:
		img = cv2.resize(load_img(img_name), (230, 230))
		cv2.imshow("real_img", img)		
	

		for i, model in enumerate(models):
			cv2.imshow(model_names[i][:10]+model_names[i][-5:-3], cv2.resize(model.predict(np.array([img]))[0], (240, 240)))

		cv2.waitKey(0)


def process_img(img):
	img = cv2.resize(img, (230, 230))
	img = img[np.newaxis, :]
	return img

def load_img(url):
	''' 
	This function takes in a url of an image and returns the image 
	'''
	return cv2.imread(url)

def show(img):
	'''
	This function displays the image passed to it
	'''
	cv2.imshow("image", img)
	cv2.waitKey(0)




if __name__ == "__main__":
	main(sys.argv)
