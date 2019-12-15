import os
import random
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras
import models
import generator

model = models.mixed_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# model = load_model("skinmodel.h5")

mc = keras.callbacks.ModelCheckpoint('model4at{epoch:08d}.h5', period=2)

model.fit_generator(generator.Data_Gen("../P_data",(250,4279), 64), epochs=20, verbose=1,validation_data=generator.Data_Gen("../P_data",(0,250),50), callbacks=[mc])

model.save("skinmodel4.h5")