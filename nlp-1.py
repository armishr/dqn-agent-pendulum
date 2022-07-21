# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:57:27 2022

@author: Adithya Raj Mishra
"""

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
#nltk.download('gutenberg')
#nltk.download('punkt')
#nltk.download('stopwords')

import string

import gensim
from gensim.models.phrases import Phraser, Phrases
from gensim.models.word2vec import Word2Vec

from sklearn.manifold import TSNE

import pandas as pd
from bokeh.io import output_notebook, output_file
from bokeh.plotting import show, figure

from nltk.corpus import gutenberg
#print(len(gutenberg.fileids()))
#print(gutenberg.fileids())

'''gberg_sents=gutenberg.sents()
word_sents=[]
for w in gberg_sents:
    word_sents.append([word.lower() for word in w if word.lower() not in list(string.punctuation)] )
    
bigram=Phraser(Phrases(word_sents,min_count=32,threshold=64))
clean_sents=[]
for w in word_sents:
    clean_sents.append(bigram[w])
    
model=Word2Vec(sentences=clean_sents,vector_size=64,sg=1,window=10,epochs=20,min_count=10,workers=4)

model.save('clean_gutenberg_model.w2v')
'''
model=gensim.models.Word2Vec.load('clean_gutenberg_model.w2v')

tsne=TSNE(n_components=2,n_iter=1000)
X_2d=tsne.fit_transform(model.wv[model.wv.key_to_index])
coords_df=pd.DataFrame(X_2d,columns=['x','y'])
coords_df['token']=model.wv.index_to_key
_=coords_df.plot.scatter('x','y',figsize=(12,12),marker='.',s=10,alpha=0.2)
print(coords_df.head())
