# -*- coding: utf-8 -*-
"""
Created on Sat May 28 14:27:41 2022

@author: Adithya Raj Mishra
"""

from tensorflow import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Embedding
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import SpatialDropout1D

#output directory
output_dir = 'model_output/conv'

#training
epochs = 4
batch_size = 128

#vector space embed
n_dim = 64
n_unique_words = 5000
max_review_length = 400
pad_type = trunc_type = 'pre'
drop_embed = 0.2

#conv layer arch
n_conv = 256
k_conv = 3

#dense layer arch
n_dense = 256
dropout = 0.2

model=Sequential()

model.add(Embedding(n_unique_words,n_dim,input_length=max_review_length))
model.add(SpatialDropout1D(drop_embed))

model.add(Conv1D(n_conv, k_conv, activation='relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(n_dense, activation= 'relu'))
model.add(Dropout(dropout))

model.add(Dense(1,activation='sigmoid'))
model.summary()

(x_train,y_train),(x_valid,y_valid) = imdb.load_data(num_words=n_unique_words)

x_train=pad_sequences(x_train,maxlen=max_review_length,padding=pad_type,truncating=trunc_type,value=0)
x_valid=pad_sequences(x_valid,maxlen=max_review_length,padding=pad_type,truncating=trunc_type,value=0)

#model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
modelcheckpoint=ModelCheckpoint(filepath = output_dir+"/weights.{epoch:02d}.hdf5")
if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)
    
#model.fit(x_train,y_train,batch_size,epochs,validation_data=(x_valid,y_valid),verbose=1,callbacks=[modelcheckpoint])
'''y_hat=model.predict(y_valid)
y_float=[]
for y in y_valid:
    y_float.append(y[0])
pct_auc=roc_auc_score(y_float, y_hat)
print(pct_auc)'''
model.load_weights(output_dir+"/weights.03.hdf5")
y_hat=model.predict(x_valid)

pct_auc=roc_auc_score(y_valid, y_hat)*100
print("{:0.2f}".format(pct_auc))
plt.hist(y_hat)