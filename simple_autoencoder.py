# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:18:14 2017

简单自动编码器：全连接的自动编码器

Keras中文手册：
自动编码器：各种各样的自动编码器
http://keras-cn.readthedocs.io/en/latest/blog/autoencoder/#_1

原文：
http://blog.keras.io/building-autoencoders-in-keras.html

@author: Ivy
"""

from keras.layers import Input, Dense
from keras.models import Model #使用的是泛化模型，注意语句格式于Sequential不同

# 搭建模型
# ===============================================================================
# 构造编码层 和 解码层：
encoding_dim = 32

input_img = Input(shape=(784,)) #图像大小28×28=784
encoded = Dense(encoding_dim, activation='relu')(input_img)  #编码：对输入进行编码
decoded = Dense(784, activation='sigmoid')(encoded)  #解码：对编码层进行解码

# 构造自动编码器：
autoencoder = Model(input=input_img, output=decoded)

# 也可以构建独立的编码器 和 解码器：
encoder = Model(input=input_img, output=encoded)

encoded_input = Input(shape=(encoding_dim,)) #编码输入
decoder_layer = autoencoder.layers[-1] #?why
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

# 编译autoencoder：
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# 训练自动编码器： 使用MNIST数据（归一化、向量化），损失函数：交叉熵（逐像素），优化器：adam
# ===============================================================================
#from keras.datasets import mnist
import numpy as np

# (x_train,_), (x_test,_) = mnist.load_data()
# 取消自动下载数据包，手动下载到当地文件夹，然后读取
# 下载地址：https://s3.amazonaws.com/img-datasets/mnist.npz
f = np.load('mnist.npz')
x_train = f['x_train']
y_train = f['y_train'] #60000
x_test = f['x_test']
y_test = f['y_test'] #10000
f.close()

x_train = x_train.astype('float32')/255.
x_test  = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test  = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  

print x_train.shape #(60000L, 784L)
print x_test.shape #(10000L, 784L)

# 拟合网络：
autoencoder.fit(x_train, x_train, #自动编码器实际上是自监督的监督学习（标签产生自其输入数据）
                nb_epoch=50,
                batch_size=256,
                shuffle=True, #大乱数据；默认True
                validation_data=(x_test,x_test))

# 重构输出，并可视化
# ===============================================================================
import matplotlib.pyplot as plt

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
#from keras.utils.visualize_util import plot #版本1的模块
from keras.utils.vis_utils import plot_model #版本2 更新了接口
plot_model(autoencoder, to_file='simple_autoencoder.png')
