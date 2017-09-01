'''
《TensorFlow官网教程》
第一章 简介
练习P10
'''

import tensorflow as tf
import numpy as np

# create python data, 100 dots
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300


# create a linear model
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Initialization
init = tf.global_variables_initializer() # tf.initialize_all_variables()

# Start a graph
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201): #xrange() was renamed to range() in Python 3
    sess.run(train)
    if step % 20 ==0:
        print(step, sess.run(W), sess.run(b))
        #输出20的倍数步时的W和b的值。

