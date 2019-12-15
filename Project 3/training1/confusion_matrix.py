'''
Ethan Pullen & Dhruv Joshi
models.py
'''
import numpy as np
import generator
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

#Confution Matrix and Classification Report
# model = load_model("../training4/model4at00000008.h5")
model = load_model("../training5/skinmodel5.h5")
validation_generator = generator.Data_Gen("../P_data",(0,250), 250)

x_batch, y_batch = validation_generator.__getitem__(0)

y_pred = model.predict(x_batch)

y_pred = np.round(y_pred)

print("pred shapes:", y_pred.shape)
print("GT shapes:", y_batch.shape)
print("ypred",y_pred)
print("validation set",y_batch)
print('Confusion Matrix')
print(confusion_matrix(y_batch, y_pred))
print('Classification Report')
target_names = ['Benign', 'Malignant']
print(classification_report(y_batch, y_pred, target_names=target_names))



