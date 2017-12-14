# -*- coding: utf-8 -*-
"""
《TensorFlow实战Google深度学习框架》
Charpter 6
这里的CNN结构参考LeNet-5，但不同的时是输入为28*28，并用了全0填充。
输入层→卷积层→池化层→卷积层→池化层→全连接层→全连接层→输出层
_________________________________________
TF_Google_charpter6_103.py：实现inference， 定义CNN:LeNet-5前向传播过程+ 神经网络中的参数
TF_Google_charpter6_203.py：train 神经网络的训练过程
TF_Google_charpter6_303.py：test 神经网络的测试过程
__________________________________
此程序进行CNN的前向传播
@author:github.com/Euniceu
"""
import tensorflow as tf

# NN parameters
Input_node = 784 #Mnist存储格式即用的列向量来存储而非二维矩阵, 在训练时输入数据要调整矩阵形状
Output_node = 10

Image_size = 28 #28*28=784
Num_channels = 1 #Mnist是灰度图
Num_labels = 10 #十分类问题

# Convolutional layer 1 卷积大小和深度
Conv1_size = 5
Conv1_deep = 32
# Convolutional layer 2
Conv2_size = 5
Conv2_deep =64
# Fully conntected layer
Fc_size = 512

# Dropout比例
dropout_rate = 0.5

# CNN的前向传播过程，train用来区分训练或测试，训练时使用dropout来提升模型可靠性、防止过拟合
def inference(input, train, regularizer):
    # 使用不同的命名空间来区分不同层的变量，不用担心重名问题
    with tf.variable_scope('layer1-conv1'):
        conv1_weight =  tf.get_variable('weight', [Conv1_size, Conv1_size, Num_channels, Conv1_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable('bias', [Conv1_deep],
                                     initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input, conv1_weight, strides=[1,1,1,1], padding='SAME')
        conv1_activ = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(conv1_activ, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        #ksize池化层过滤器的大小，第一、四维必须是1：过滤器不可以跨不同输入样例、节点矩阵深度
        #strides步长，第一、四维必须是1：池化层不能改变节点矩阵的深度、输入样例的个数

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable('weight',[Conv2_size, Conv2_size, Conv1_deep, Conv2_deep],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable('bias', [Conv2_deep],
                                     initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1,1,1,1], padding='SAME')
        conv2_activ = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(conv2_activ, [1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # 输入全连接层前要将卷积矩阵摊平
        pool_shape = pool2.get_shape().as_list() #获取输出矩阵的维度 结果是四个维度，D1是一个batch内的数据数，D2-4是长、宽、深
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] #摊平后的长度=矩阵的长*宽*深
        pool2_flatten = tf.reshape(pool2, [pool_shape[0], nodes]) #摊平操作 即 reshape

    with tf.variable_scope('layer5-fc1'):
        fc1_weight = tf.get_variable('weight', [nodes, Fc_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重进行正则化！
        if regularizer is not None:
            tf.add_to_collection("loss_regularize", regularizer(fc1_weight))

        fc1_bias = tf.get_variable('bias', [Fc_size],
                                  initializer=tf.constant_initializer(0.1))
        fc1 = tf.matmul(pool2_flatten, fc1_weight) + fc1_bias
        fc1_activ = tf.nn.relu(fc1)
        # 在训练过程中加入dropout
        if train:
            fc1_activ = tf.nn.dropout(fc1_activ, dropout_rate)

    with tf.variable_scope('layer5-fc2'):
        fc2_weight = tf.get_variable('weight', [Fc_size, Output_node],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection("loss_regularize", regularizer(fc2_weight))

        fc2_bias = tf.get_variable('bias', [Output_node],
                                   initializer=tf.constant_initializer(0.1))

        fc2 = tf.matmul(fc1_activ, fc2_weight) + fc2_bias
        # 最后一层不要激活，因为最后一层的输出用softmax分类器，在使用softmax_cross_entropy时会自动使用softmax计算
        logit = fc2

    return logit



