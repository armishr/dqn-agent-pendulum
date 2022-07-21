# -*- coding: utf-8 -*-
"""
Created on Thu May 12 23:43:40 2022

@author: Adithya Raj Mishra
"""

from tensorflow import keras
import tensorflow as tf
from keras.datasets import mnist,boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt
from keras.layers.normalization import batch_normalization
from keras.callbacks import TensorBoard
import numpy as np


#loading and view training and verifying data
#(X_train,y_train),(X_valid,y_valid)=mnist.load_data();
#plt.figure(figsize=(5,5))
(X_train,y_train),(X_valid,y_valid)=boston_housing.load_data()
#print(y_train[0])
"""for k in range(12):
    plt.subplot(1, 12,k+1)
    plt.imshow(X_train[k],cmap='Greys')
    plt.axis('off')
plt.tight_layout()
plt.show()
#pre-processing and formating data
X_train=X_train.reshape(60000,784).astype('float32')
X_valid=X_valid.reshape(10000,784).astype('float32')

X_train/=255
X_valid/=255
"""

#one-hot conversion
'''n_classes=10
y_train=keras.utils.to_categorical(y_train,n_classes)
y_valid=keras.utils.to_categorical(y_valid,n_classes)
#print(y_train[2])'''
tensorboard=TensorBoard('logs/deep-net')
#creating model
model=Sequential()
model.add(Dense(32,activation=tf.keras.layers.ReLU() ,input_shape=(13,)))
#model.add(batch_normalization.BatchNormalization())
model.add(Dense(64  ,activation=tf.keras.layers.ReLU()))
#model.add(batch_normalization.BatchNormalization())
model.add(Dense(64  ,activation=tf.keras.layers.ReLU()))
model.add(batch_normalization.BatchNormalization())

model.add(Dense(1,activation='linear'))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,batch_size=8,epochs=100,verbose=1,validation_data=(X_valid,y_valid),callbacks=[tensorboard])

print(model.predict(np.reshape(X_train[0],[1,13])))
print(y_train[0])