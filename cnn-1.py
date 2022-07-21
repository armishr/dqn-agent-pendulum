# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:23:33 2022

@author: Adithya Raj Mishra
"""
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.datasets import mnist

n_classes=10
(X_train,Y_train),(X_valid,Y_valid)=mnist.load_data()

Y_train=keras.utils.to_categorical(Y_train,n_classes)
Y_valid=keras.utils.to_categorical(Y_valid,n_classes)

X_train=X_train.reshape(60000,28,28,1).astype('float32')
X_valid=X_valid.reshape(10000,28,28,1).astype('float32')


model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(n_classes,activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=X_train,y=Y_train,batch_size=32,epochs=20,verbose=1,validation_data=(X_valid,Y_valid))