# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:31:15 2022

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
from keras.layers import SimpleRNN

#output directory
output_dir = 'model_output/rnn'

#training
epochs = 16
batch_size = 128

#vector space embed
n_dim = 64
n_unique_words = 10000
max_review_length = 100
pad_type = trunc_type = 'pre'
drop_embed = 0.2

#conv layer arch
n_conv = 256
k_conv = 3

#dense layer arch
n_dense = 256
dropout = 0.2

#rnn layer arch
n_rnn = 256
drop_rnn = 0.2

model=Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length = max_review_length))
model.add(SpatialDropout1D(drop_embed))
model.add(SimpleRNN(n_rnn, dropout=drop_rnn))
model.add(Dense(1,activation='sigmoid'))
model.summary()

(x_train,y_train),(x_valid,y_valid) = imdb.load_data(num_words=n_unique_words)

x_train=pad_sequences(x_train,maxlen=max_review_length,padding=pad_type,truncating=trunc_type,value=0)
x_valid=pad_sequences(x_valid,maxlen=max_review_length,padding=pad_type,truncating=trunc_type,value=0)

#model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
modelcheckpoint=ModelCheckpoint(filepath = output_dir+"/weights.{epoch:02d}.hdf5")
if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)
    
#model.fit(x_train,y_train,batch_size,epochs,verbose=1,callbacks=[modelcheckpoint],validation_data=(x_valid,y_valid))
model.load_weights(output_dir+"/weights.16.hdf5")
y_hat=model.predict(x_valid)
pct_auc=roc_auc_score(y_valid, y_hat)*100
print(pct_auc)