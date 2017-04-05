# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 20:29:54 2017

去噪自动编码器：使用自动编码器进行图像去噪
手动给图片加入噪声，解码器提纯

输出可视化，
模型可视化

Keras中文手册：
自动编码器：各种各样的自动编码器
http://keras-cn.readthedocs.io/en/latest/blog/autoencoder/

原文：
http://blog.keras.io/building-autoencoders-in-keras.html
@author: Ivy
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
#===============================================================================
# 数据
f = np.load('mnist.npz')
x_train = f['x_train']
y_train = f['y_train'] #60000 样本个数
x_test = f['x_test']
y_test = f['y_test'] #10000
f.close()

x_train = x_train.astype('float32')/255.
x_test  = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train), 28,28,1))  #处理成图片格式
x_test  = np.reshape(x_test,  (len(x_test), 28,28,1))

print x_train.shape #(60000L, 28L，28L，1L) 区别于一维数据的(60000L, 784L)
print x_test.shape #(10000L, 28L, 28L, 1L)

# 加噪
noise_factor = 0.5 #噪音比例因子
x_train_noisy = x_train + \
                noise_factor*np.random.normal(loc=0.0, scale=1.0,size=x_train.shape) #白噪声？
x_test_noisy = x_test + noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#n = 10
#plt.figure()#figsize=(20,4))
#for i in range(n):
#    #原图
#    ax = plt.subplot(2,n,i+1) 
#    plt.imshow(x_test[i].reshape(28,28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    
#    ax = plt.subplot(2,n,i+1+n)
#    plt.imshow(x_test_noisy[i].reshape(28,28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()

# 降噪自动编码器
input_img = Input(shape=(28,28,1)) #‘th’模式需要
x = Conv2D(32,(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(32,(3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(32,(3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(32,(3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3), activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train, #训练数据是含噪图像，标签是纯净图像
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))#,
#                callbacks=[TensorBoard(log_dir='/tmp/tb',histogram_freq=0,
#                 write_graph=False)])

# 显示原始图片 和 被加噪后的图片 和 去噪后的图像
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure()#figsize=(20,4))
for i in range(n):
    #原图
    ax = plt.subplot(3,n,i+1) 
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3,n,i+1+n)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3,n,i+1+n+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()