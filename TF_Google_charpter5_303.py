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
此程序进行神经网络测试

@author:github.com/Euniceu
"""

import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import TF_Google_charpter5_103 as fwpropagation
import TF_Google_charpter5_203 as NNtrain

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, fwpropagation.Input_node], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, fwpropagation.Output_node], name="y-iutput")
        validation_feed ={x:mnist.validation.images,
                          y_:mnist.validation.labels}

        y = fwpropagation.inference(x, None) #测试时不关注正则化损失值

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名来加载模型
        variable_average = tf.train.ExponentialMovingAverage(NNtrain.Moving_average_decay)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # ?!
        # 每隔10s调用一次计算正确率的过程以检测训练过程中的正确率的变化
        while True:
            with tf.Session as sess:
                ckpt = tf.train.get_checkpoint_state(NNtrain.Model_save_path)
                #自动找到目录中最新模型的文件名

                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    accureacy_score = sess.run(accuracy, feed_dict=validation_feed)
                    print("After %s training step(s), validation accuracy = % g" %
                          (global_step, accureacy_score))
                else:
                    print("NO checkpoint found")
                    return
                # 推迟10s执行，调用一次计算正确率的过程来检测训练过程中的正确率变化
                time.sleep(10)

def main(argv=None):
    mnist = input_data.read_data_sets("/data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
