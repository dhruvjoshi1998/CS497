'''
Ethan Pullen & Dhruv Joshi
models.py
'''
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, concatenate, Flatten

NUM_PIPELINES = 2


def make_cnn():
	model = Sequential()
	model.add(Conv2D(16, kernel_size = (15, 15), strides = 2, activation='relu', input_shape = (768, 1024, 3)))
	model.add(Conv2D(16, kernel_size = (7, 7), strides = 2, activation='relu'))
	for _ in range(NUM_PIPELINES):
		model.add(Conv2D(16, kernel_size = (3, 3), activation='relu'))
		# model.add(Conv2D(16, kernel_size = (5, 5), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(rate=0.125))
	model.add(MaxPooling2D(pool_size=(8, 8)))
	model.add(Flatten())
	model.add(Dense(54, activation = 'relu'))
	return model

def make_mlp():
	model = Sequential()
	model.add(Dense(10, activation = 'relu', input_shape = (2,)))
	model.add(Dense(10, activation = 'relu'))
	return model

def mixed_model():
	mlp = make_mlp()
	cnn = make_cnn()

	cout = concatenate([mlp.output, cnn.output])

	x = Dense(16, activation = 'relu')(cout)
	x = Dense(16, activation = 'relu')(x)
	x = Dense(16, activation = 'relu')(x)
	x = Dense(16, activation = 'relu')(x)
	x = Dense(1, activation = 'sigmoid')(x)

	model = Model([mlp.input,cnn.input], x)

	return model
