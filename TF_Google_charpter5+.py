# -*- coding: utf-8 -*-
"""
《TensorFlow实战Google深度学习框架》
Charpter 5
MNIST数字识别问题
目的：验证使用滑动平均模型的准确率

优化：使用tf.variable和tf.get_variable
在使用inference时不再需要输入神经网络的所有参数

@author:github.com/Euniceu
"""

import tensorflow as tf

# MNIST dataset
Input_node = 784 #28*28
Output_node = 10 #0~9 digits

# Neural Network
Layer1_node = 500 #Hidden layer 1
batch_size = 100
Learning_rate_base = 0.8 #基础学习率
Learning_rate_decay = 0.99 #学习率衰减率
Regularization_rate = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
Training_steps = 30000 #训练轮数
Moving_average_decay = 0.99 #滑动平均衰减

# 输入输出
x = tf.placeholder(tf.float32, [None, Input_node], name='x-input')
y_ = tf.placeholder(tf.float32, [None, Output_node], name='y-input')
# 参数
w1 = tf.Variable(tf.truncated_normal([Input_node, Layer1_node], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[Layer1_node]))
w2 = tf.Variable(tf.truncated_normal([Layer1_node, Output_node], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[Output_node]))


# 定义辅助函数，给定神经网络输入和参数，计算前向传播结果。
# 两层神经网络，考虑了滑动平均模型
def inference(input, avg_class, reuse=False):
    # 不使用滑动平均：
    if avg_class == None:
        with tf.variable_scope('layer1', reuse=reuse):
            weight = tf.get_variable("weight", [Input_node, Layer1_node],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("bias", [Layer1_node],
                                   initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input, weight) + bias)

        with tf.variable_scope('layer2', reuse=reuse):
            weight = tf.get_variable("weight", [Layer1_node, Output_node],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("bias", [Output_node],
                                   initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, weight) + bias
            return layer2
    else:
        with tf.variable_scope('layer1', reuse=True):
            weight = tf.get_variable("weight", [Input_node, Layer1_node],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable('bias', [Layer1_node],
                                   initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input, avg_class.average(weight)) +
                                      avg_class.average(bias))
        with tf.variable_scope('layer2', reuse=True):
            weight = tf.get_variable("weight", [Layer1_node, Output_node],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("bias", [Output_node],
                                   initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, avg_class.average(weight))+avg_class.average(bias)
        return layer2
# Train
def train(mnist):
    # 不使用滑动平均的前向传播结果
    y = inference(x, None, True)

    global_step = tf.Variable(0, trainable=False) #存储训练轮数，指定为不可训练参数

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_average = tf.train.ExponentialMovingAverage(MoJving_average_decay, global_step)

    # 滑动平均操作，应用于可训练变量（能代表NN参数的变量，其他变量设置为trainable=False）
    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 使用滑动平均后的前向传播结果
    y_average = inference(x, variable_average, True)

    # 训练不使用滑动平均模型的NN，然后同时更新它的参数和每个参数的滑动平均值，
    # 最后计算出滑动平均后与真实标签的相符程度即正确率
    # ——————————————————————————————————————————————————————————————————————————————————————
    # 计算损失：
    # ————————————
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(y_, 1),
            logits=y)
        #给出的y_标签是一个长度为10的一维数组，但cross_entropy函数需要的是一个数字，
        #因此用tf.argmax()来得到正确答案对应的标签
    cross_entropy_mean = tf.reduce_mean(cross_entropy) #当前batch的交叉熵平均值

    # L2正则化损失
    regularizer = tf.contrib.layers.l2_regularizer(Regularization_rate)
    regularization = regularizer(w1) + regularizer(w2) #一般只计算权重的正则化损失，不计算偏置的

    loss = cross_entropy_mean + regularization

    # 指数衰减学习率：
    # ————————————————
    learning_rate = tf.train.exponential_decay(
        Learning_rate_base, global_step,
        mnist.train.num_examples/batch_size, Learning_rate_decay)

    # 优化算法优化损失函数：
    # ————————————————————————
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step = global_step)

    # 训练更新步
    # 可以同时更新神经网络参数 和 每一个参数的滑动平均值
    train_op = tf.group(train_step, variable_average_op)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(y_average, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 运行
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 验证数据集
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
        # 测试数据集
        test_feed = {x:mnist.test.images, y_:mnist.test.labels}

        # 迭代训练神经网络：
        Valid_ACC = []
        Test_ACC = []
        for i in range(Training_steps):
           if i % 1000 == 0: #每1000轮输出一次在验证集上的测试结果
               validate_acc = sess.run(accuracy, feed_dict=validate_feed)
               test_acc = sess.run(accuracy, feed_dict=test_feed)
               print("After %d training step(s), validation accuracy"
                  "using average model is %g, test accuracy unsing average "
                  "model is %g " % (i, validate_acc, test_acc))
               Valid_ACC.append(validate_acc)
               Test_ACC.append(test_acc)


           xs, ys = mnist.train.next_batch(batch_size)
           sess.run(train_op, feed_dict={x:xs, y_:ys})

        # 训练结束后，在测试集上检测神经网络模型的正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average "
             "model is %g " % (Training_steps, test_acc))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(Valid_ACC)
        plt.plot(Test_ACC)
        plt.show()


# main program
def main(argv=None):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/data", one_hot=True)
    train(mnist)

# 主程序入口，会调用上面定义的main
if __name__ == '__main__':
    tf.app.run()