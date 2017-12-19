# -*- coding: utf-8 -*-
"""
《TensorFlow实战Google深度学习框架》
Charpter 8
用LSTM来预测sin函数值
@author:github.com/Euniceu
"""
import numpy as np
import tensorflow as tf
import matplotlib as mpl

mpl.use('Agg') #没有GUI时使用matplotlib绘图,必须添加在import matplotlib.pyplot之前，否则无效
import matplotlib.pyplot as plt

learn = tf.contrib.learn #使用TFLearn框架

Hidden_size = 30
Num_layers = 2 #LSTM的层数
Timesteps = 10 #RNN的截断长度
Training_steps = 10000 #训练轮数
Batch_size = 32
Training_examples = 10000 #训练数据个数
Testing_examples = 100 #测试集数据个数
Sample_gap = 0.01 #采样间隔

def generate_data(seq):
    x = []
    y = []
    for i in range(len(seq)-Timesteps-1):
        x.append([seq[i:i+Timesteps]]) #用sin函数前面的Timesteps个点的信息作为输入
        y.append([seq[i+Timesteps]]) #来预测第i+Timesteps个点的函数值（输出）
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(x, y):
    # 多层lstm结构
    # tensorflow升级后改变
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(Hidden_size)
    #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * Num_layers)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(Hidden_size)
    #cell = tf.contrib.rnn.MultiRNNCell([[lstm_cell()] for _ in range(Num_layers)])
    #cell = tf.contrib.rnn.MultiRNNCell([lstm_cell()]*Num_layers)
    cells=[]
    # ?????  有错误的？？？？
    for i in range(Num_layers):
        cells.append(lstm_cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    #cell = tf.contrib.rnn.MultiRNNCell(input([lstm_cell() for _ in range(Num_layers)]), state_is_tuple=True)
    #rnn.MultiRNNCell.__call__
    #x_ = tf.unpack(x, axis = 1)
    x_ = tf.unstack(x, axis = 1)

    # 讲多层LSTM结构连接成RNN网络 并 计算前向传播结果
    #output, _  = tf.nn.rnn(cell, x_, dtype=tf.float32)
    output, _ = tf.contrib.rnn.static_rnn(cell, x_, dtype=tf.float32)
    output = output[-1] #这里只关注最后一个时刻的输出结果

    prediction, loss =learn.models.linear_regression(output, y)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer = 'Adagrad', learning_rate=0.1)

    return prediction, loss, train_op

# _______ 建立深层循环模型 ————————————————————————————————————————
regressor = learn.Estimator(model_fn = lstm_model)

# 产生训练 和 测试数据集合，用正弦函数生成
test_start = Training_examples * Sample_gap
test_end = (Training_examples + Testing_examples) * Sample_gap
train_x, train_y = generate_data(np.sin(np.linspace(0, test_start, Training_examples, dtype=np.float32)))
#np.linespace(a,b,l) 从a到b产生长度为l的等差序列数组
test_x, test_y = generate_data(np.sin(np.linspace(test_start, test_end, Testing_examples,dtype=np.float32)))

# fit训练模型
regressor.fit(train_x, train_y, batch_size=Batch_size, steps=Training_steps)

# 对测试数据进行预测
predicted = [[pred] for pred in regressor.predict(test_x)]
rmse = np.sqrt(((predicted-test_y)**2).mean(axis=0))
print("Mean Square Error is : %f" % rmse[0])

# 画图
fig = plt.figure()
plot_predicted = plt.plot(predicted, label='predicted')
plot_test = plt.plot(test_y, label='reas_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
fig.savefig('sin.png')



