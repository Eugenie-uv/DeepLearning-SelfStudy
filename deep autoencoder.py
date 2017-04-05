# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:09:02 2017

深度自动编码器：叠加多个自动编码器

输出可视化，
模型可视化

Keras中文手册：
自动编码器：各种各样的自动编码器
http://keras-cn.readthedocs.io/en/latest/blog/autoencoder/

原文：
http://blog.keras.io/building-autoencoders-in-keras.html
@author: Ivy
"""

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt #结果/输出可视化
from keras.utils.vis_utils import plot_model #模型可视化，保存成png文件
#===============================================================================

# 模型搭建
input_img = Input(shape=(784,)) #28*28=784 列/特征
encoded = Dense(128, activation='relu')(input_img)
encoded1 = encoded
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded) #为什么是32，而不是28？

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
#decoded2 = decoded
decoded = Dense(784, activation='relu')(decoded) 
decoded3 = decoded
autoencoder = Model(inputs=input_img, outputs=decoded)

# ---模型可视化用到的独立编码层和解码层--------------------------------------------------------
encoder = Model(inputs=input_img, outputs=encoded1) #第一层编码器，用于可视化输入图片
# keras2将Model里的关键词input、output改为inputs、output
# ？
#encoding_dim = 32
#encoded_input = Input(shape=(encoding_dim,)) #编码输入
#decoded_input = Input(shape=(64,))
decoded_input = Input(shape=(128,)) #倒数第二层的大小，作为输入输入到最后一层
decoder_layer = autoencoder.layers[-1] #最后一层，即想要可视化的输出
decoder = Model(inputs=decoded_input, outputs=decoder_layer(decoded_input))
#decoder = Model(input=np.get_output_at(autoencoder.layers[-1]), output=decoder_layer(autoencoder.layers[-1].output))

# 模型编译
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# 数据
f = np.load('mnist.npz')
x_train = f['x_train']
y_train = f['y_train'] #60000 样本个数
x_test = f['x_test']
y_test = f['y_test'] #10000
f.close()

x_train = x_train.astype('float32')/255.
x_test  = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test  = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  

print x_train.shape #(60000L, 784L)
print x_test.shape #(10000L, 784L)

# 模型拟合
autoencoder.fit(x_train, x_train,
                epochs=100,  # keras2改nb_epoch 为 epochs
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# 输出可视化
# --------------------------------------------------
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n =10 # 要可视化的数字图片个数
plt.figure(figsize=(20,4))
for i in range(n):
    # 原图：
    ax = plt.subplot(2, n, i+1) #!这里必须用i+1，而不是i，因为ValueError: num must be 1 <= num <= 20, not 0
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()  #显示灰度图片
    ax.get_xaxis().set_visible(False) #不显示x轴坐标
    ax.get_yaxis().set_visible(False) #不显示y轴坐标
    
    # 重构图：
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# 模型可视化
# ==============================================================================
plot_model(autoencoder, to_file='deep_autoencoder.png')