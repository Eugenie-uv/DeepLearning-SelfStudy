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
____________________________________
此程序中定义的神经网络的前向传播算法，无论训练还是测试，都可以直接调用inference函数，
不用关心具体的神经网络结构。

@author:github.com/Euniceu
"""
import tensorflow as tf

# 神经网络结构参数
Input_node = 784
Output_node = 10
Layer1_node = 500

def get_weight_variable(shape, regularizer):
    weight = tf.get_variable("weight", [Input_node, Layer1_node],
                             initializer=tf.truncated_normal_initializer()) #通过变量名称来使用变量 不仅仅是单纯创建变量tf.Variable
    if regularizer != None:
        tf.add_to_collection("loss_regularize", regularizer(weight))
    return weight

def inference(input, regularizer):
    with tf.variable_scope("layer1"):
        weight = get_weight_variable([Input_node, Layer1_node], regularizer)
        bias = tf.get_variable("bias", [Layer1_node],
                               initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input, weight) + bias)

    with tf.variable_scope("layer2"):
        weight = get_weight_variable([Layer1_node, Output_node], regularizer)
        bias = tf.get_variable("bias", [Output_node],
                               initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weight)+bias
    return layer2

