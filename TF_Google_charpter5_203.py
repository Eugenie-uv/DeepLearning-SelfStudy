# -*- coding: utf-8 -*-
"""
《TensorFlow实战Google深度学习框架》
Charpter 5
优化编码，分成了3个程序，使用更加灵活
TF_Google_charpter5_103.py：实现inference，
定义前向传播过程+神经网络中的参数

TF_Google_charpter5_203.py：train
神经网络的训练过程

TF_Google_charpter5_303.py：test
神经网络的测试过程
___________________________________
此程序进行神经网络训练

@author:github.com/Euniceu
"""
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import TF_Google_charpter5_103 as fwpropagation

# 神经网络的参数
Batch_size = 100
Learning_rate_base = 0.8
Learning_rate_decay = 0.99
uRegularaztion_rate = 0.0001
Traing_steps = 30000
Moving_average_decay = 0.99

# 模型保存的路径和文件名
Model_save_path = "/path/to/model/"
Model_name = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, [None, fwpropagation.Input_node], name="x-input")
    y_ = tf.placeholder(tf.float32, [fwpropagation.Output_node], name="y-input")

    # 使用L2正则化
    regularizer = tf.contrib.layers.l2_regularizer(Regularaztion_rate)

    y = fwpropagation.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("loss_regularize"))

    # 指数衰减学习率
    Learning_rate = tf.train.exponential_decay(
        Learning_rate_base,
        global_step,
        mnist.train.num_examples/Batch_size,
        Learning_rate_decay)

    # 训练更新步
    train_step = tf.train.GradientDescentOptimizer(Learning_rate).minimize(loss, global_step=global_step)

    # 移动平均模型
    variable_average = tf.train.ExponentialMovingAverage(Moving_average_decay, global_step=global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 同时更新训练步+移动平均操作
    train_op = tf.group(train_step, variable_average_op)

    # 初始化Tensorflow持久化
    saver = tf.train.Saver()
    # 运行图
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(Traing_steps):
            xs, ys = mnist.train.next_batch(Batch_size)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x:xs, y_:ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is % g" %
                      (step, loss_value))
                saver.save(sess, os.path.join(Model_save_path, Model_name), global_step=global_step)

    def main(argv=None):
        mnist = input_data.read_data_sets("/data", one_hot=True)
        train(mnist)

    if __name__ == '__main__':
        tf.app.run()
