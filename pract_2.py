'''
《TensorFlow官网教程》
第2章 MNIST
练习P14
一层Softmax Regression模型来进行MINST手写数字识别
'''

# import mnist dataset:
from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder = "F:\DATA\MNIST_data" #数据存放路径
mnist = input_data.read_data_sets("MNIST_data_folder",one_hot=True)

# BUILD THE GRAPH
# ===============================================
import tensorflow as tf

# Softmax Regression Model
# input & output:
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# model reference:
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Softmax model:
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cost function: cross entry
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# TRAIN THE MODEL
# ==============================================
# Intalization the Variables:
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Optimize means to update the regerences:
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x:batch[0], y_:batch[1]}) #train_step.run(feed_dict={x:batch[0], y_:batch[1]})

# Evaluate the Model
# ================================================
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x:mnist.train.images, y_:mnist.train.labels}))
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))