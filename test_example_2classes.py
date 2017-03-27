# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:59:56 2017
Keras 1.0 中文文档 P20 训练例子——二分类
@author: Ivy
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np

#for a single-input model with 2 classes(binary):
model = Sequential()
model.add(Dense(1, input_dim=784, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#generate dumpy data:
data = np.random.random((100,784))
labels = np.random.randint(2,size=(1000,1))

#train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, nb_epoch=1000, batch_size=32)