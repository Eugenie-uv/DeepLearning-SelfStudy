# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:37:00 2017
Keras 1.0 中文文档 P20 训练例子——十分类
@author: Ivy
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge
import numpy as np
from keras.utils.np_utils import to_categorical

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784)) # 32是一个图像的大小维度

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch,right_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax')) # 输出是10个分类

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#generate dumpy data
data_1 = np.random.random((1000,784))
data_2 = np.random.random((1000,784))

labels = np.random.randint(10,size=(1000,1))
# 将一列的标签按标签值分列，变成10列矩阵，每一列是统一标签值
labels = to_categorical(labels, 10)

#train the model
#the model has 2 inputs
model.fit([data_1,data_2], labels, nb_epoch=10, batch_size=32)