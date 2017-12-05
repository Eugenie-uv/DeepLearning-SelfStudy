# -*- coding: utf-8 -*-
"""

@author:github.com/Euniceu
"""
# _____________ 自定义损失函数 ________________________________________________
'''
# 自定义损失函数示例 单层神经网络（没有隐藏层）= 感知机

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 输入输出
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 模型参数
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

# 网络模型
# no hidden layer
y =  tf.matmul(x,w) #全连接网络

# 自定义损失函数 成本1和利润10不同→侧重率不同:生产少1，少赚10；生产多1，少挣1
# 最大化利润→最小化成本/代价
loss_less = 10 #y预测少了<y_，代价就是10
loss_more = 1 #y预测多了，代价为1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                               (y - y_)* loss_more, #True:y>y_
                               (y_ - y)* loss_less)) #False:y<y_
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 生成模拟数据集及真值标签
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 标签就设为两个输入的和+一个随机量/噪声
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2 ) in X]

# Train
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("weight at start:")
    print(sess.run(w))
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
    print("weight at end:")
    print(sess.run(w))

'''

# _____________ 集合collection收集L2正则化损失 ________________________________________________
'''
# 5层神经网络+L2正则化的损失函数（通过“集合collection”计算）
import tensorflow as tf

def get_weight(shape, lambda):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lambda)(var))
    #将var的L2规则化损失以lamda比重加入losses集合中
    return var

x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder((tf.float32, shape=[None, 1]))

batch_size = 8
layer_dimension = [2, 10,  10, 10, 1]
n_layers = len(layer_dimension)

cur_layer = x #当前层，管理前向传播时最深层的节点,开始时是输入层
in_demension = layer_dimension[0] #当前层的节点数
# for循环生成5层全连接神经网络
for i in range(n_layers):
    out_demension = layer_dimension[i]
    # 生成权重，并保存其L2规范化
    weight = get_weight([in_demension, out_demension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_demension]))
    # 当前层网络输出 ReLu激活
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 下一层的输入=本层输出
    in_demension = layer_dimension[i]

# 此时losses里已经包含了各个网络层的L2损失项，现在加入训练数据中的体现出的损失：均方误差损失
mse_loss = tf.reduce_sum(tf.square(y_ - cur_layer))
tf.add_to_collection("losses", mse_loss)

# 得到总损失：将losses集合中的所有损失成分加起来
loss = tf.add_n(tf.get_collection("losses"))
'''

# _____________ 滑动平均模型 ________________________________________________
#'''
# 滑动平均模型
import tensorflow as tf

var_ma = tf.Variable(0, dtype=tf.float32) #用于计算滑动平均的变量
decay = 0.99 #初始衰减率
per_update = tf.Variable(0, trainable=False) #用于动态设置decay的大小

ema = tf.train.ExponentialMovingAverage(decay, per_update)
maintain_averages_op = ema.apply([var_ma]) #每次执行这个操作这个列表中的变量就会被更新

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run([var_ma, ema.average(var_ma)])) #ema.average()获得滑动平均之后的变量取值

    sess.run(tf.assign(var_ma, 5)) #给变量赋值5
    sess.run(maintain_averages_op)
    print(sess.run([var_ma, ema.average(var_ma)]))

    sess.run(tf.assign(per_update, 10000))
    sess.run(tf.assign(var_ma, 10))
    sess.run(maintain_averages_op)
    print(sess.run([var_ma, ema.average(var_ma)]))

    sess.run(maintain_averages_op) #执行一次滑动平均操作！
    print(sess.run([var_ma, ema.average(var_ma)]))
#'''

