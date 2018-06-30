#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
# 第一行注释是为了告诉Linux/OS X系统，这是一个Python可执行程序，Windows系统会忽略这个注释；
#第二行注释是为了告诉Python解释器，按照UTF-8编码读取源代码，否则，你在源代码中写的中文输出可能会有乱码
______________________
Created on 2018.6.19
快速入门深度学习五步法
_________________________
@author:github.com/Euniceu
'''

'''
# 过程式构造——add串联
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(units=54, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy')]
# 通过类调用更多参数：为随机梯度加上Nesterov动量，生成一个SGD对象
from keras.optimizer import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 评估模型
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# 用模型来预测
classes = model.predict(x_test, batch_size=128)
'''

'''
# 函数式编程：官方例子
from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.fit(data, labels)
'''

'''
# 函数式编程：有并联的模型
from keras.layers import Conv2D, MaxPooling2D, Input
input_img = Input(shape=(256, 256, 3))
tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)

output = kera.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
'''

'''
CNN处理MNIST手写数字识别
模型是线性的，使用Sequential容器
'''
# from __future__ import print_function
import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D
from keras import backend as K

# 参数设置
batch_size = 128
num_classes = 10 #十分类
epochs = 12

img_rows, img_cols = 28, 28

# 数据集
(x_train, y_train), (x_test. y_test) = mnist.load_data()
if K.image_data_format() == 'chaneels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 变量设置
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 搭建模型
model = Sequential()

# 卷积层
model.add(Conv2D(32, kernel_size=(3,3),
        activation='relu',
        input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))

# 池化层：防止过拟合
model.add(MaxPooling2D(pool_size=(2,2)))

# Dropout
model.add(Dropout(0.25))

# 全连接层：要先摊平，Flatten层
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Softmax输出
model.add(Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizer.Adadelta(),
        metrics=['accuracy'])

# 运行
model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        balidation_data=(x_test. y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])