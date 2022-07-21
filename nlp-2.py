# -*- coding: utf-8 -*-
"""
Created on Fri May 27 23:33:54 2022

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

#output-dir
output_dir = 'model_output/dense'

#training
epochs = 4
batch_size = 128

#vec-space embeds
n_dim = 64
n_unique_words = 5000
n_words_to_skip = 50
max_review_length = 100
pad_type = trunc_type = 'pre'

#neural network arch
n_dense = 64
dropout = 0.5

#(all_x_train,_),(all_x_valid,_)=imdb.load_data()
(x_train,y_train),(x_valid,y_valid) = imdb.load_data(num_words=n_unique_words,skip_top=n_words_to_skip)

"""word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['PAD'] = 0        
word_index['START'] = 1
word_index['UNK'] = 2
index_word={v:k for k,v in word_index.items()}
print(' '.join(index_word[ids] for ids in x_train[0])+'\n'+' '.join(index_word[ids] for ids in all_x_train[0]))
"""
#print(len(x_train[5]))
x_train=pad_sequences(x_train,maxlen=max_review_length,padding=pad_type,truncating=trunc_type,value=0,)
x_valid=pad_sequences(x_valid,maxlen=max_review_length,padding=pad_type,truncating=trunc_type,value=0,)
#print(len(x_train[5]))
model=Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1,activation='sigmoid'))
model.summary()
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
modelcheckpoint=ModelCheckpoint(filepath = output_dir+"/weights.{epoch:02d}.hdf5")
if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)
    
#model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid,y_valid),callbacks=[modelcheckpoint])
model.load_weights(output_dir+"/weights.02.hdf5")
y_hat = model.predict(x_valid)
plt.hist(y_hat)
pct_auc = roc_auc_score(y_valid, y_hat)*100.0
print("{:0.2f}".format(pct_auc))

float_y_hat = []
for y in y_hat:
    float_y_hat.append(y[0])
ydf = pd.DataFrame(list(zip(float_y_hat,y_valid)),columns=['y_hat','y'])
#print(ydf[(ydf.y == 1) & (ydf.y_hat < 0.1)].head(10))
