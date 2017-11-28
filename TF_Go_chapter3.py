# -*- coding: utf-8 -*-
"""
《TensorFlow实战Google深度学习框架》
Charpter 3

@author:github.com/Euniceu
"""

'''
# 入门
import tensorflow as tf

# batch大小 批次
batch = 3

# 设置随机数种子seed
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#x = tf.constant([[0.7, 0.9]])
x = tf.placeholder(tf.float32, shape=(batch,2), name="input")

# 最简单的两层神经网络
# 前向传播

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 变量单个初始化
#sess.run(w1.initializer)
#sess.run(w2.initializer)

# 同时初始化所有变量
init_op = tf.initialize_all_variables()
sess.run(init_op)

#print(sess.run(y))
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
sess.close()
'''

# 例子：二分类问题
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8 #batch大小

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1)) #模型参数 两层网络就有两个参数，两层网络是指去除输入层的网络层数=隐藏层+输出层
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input') #None可以方便的使用不同的batch大小，训练时使用小batch，测试时使用全部测试数据
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# Forward propagation
h1 = tf.matmul(x, w1) #matmul:矩阵相乘，这里是全连接网络
y = tf.matmul(h1, w2)

# Loss function & Backpropagation
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# Random dataset
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X] #定义规则设置标签，x1+x2<1的为正样本1，其他为0

# Train
with tf.Session() as sess: #上下文里运行就不需要sess.close了
    init_op  = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    steps = 5000
    for i in range(steps):
        # 选取batch个样本进行训练：
        start = (i*batch_size) % dataset_size #余数即batch起点
        end = min(start+batch_size, dataset_size)

        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            # 每隔1000步输出整个数据集上的损失
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print("After %d training step(s), the cross entropy on all data is %g" % (i, total_cross_entropy))
    print("w1 at end is: ")
    print(sess.run(w1))
    print("w2 at end is: ")
    print(sess.run(w2))





