# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:02:23 2016
11行Python代码建立神经网络
练习

反向传播训练
根据输入的三栏数据X预测输出结
英文原文：http://iamtrask.github.io/2015/07/12/basic-python-network/ 

@author: Ivy
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
reload(sys)  
sys.setdefaultencoding('utf8')

# ==|简节代码|=========================
#X = np.array([[0,0,1],
#              [0,1,1],
#              [1,0,1],
#              [1,1,1]])
#y = np.array([[0,1,1,0]]).T
#np.random.seed(1)
#syn0 = 2*np.random.random((3,4))-1
#syn1 = 2*np.random.random((4,1))-1
#for j in xrange(60000):
#    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
#    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
#    l2_delta = (y-l2)*(l2*(1-l2))
#    l1_delta = l2_delta.dot(syn1.T)*(l1*(1-l1))
#    syn1 += l1.T.dot(l2_delta)
#    syn0 +=X.T.dot(l1_delta)

# ====|两层神经网络|===============================================
#sigmoid function
#将任何值映射到0-1范围内，将实数转为概率值
#优异特性之一：sigmoid函数的导数就是它的输出值乘以1-输出值：out*（1-out）
def nonlin(x,deriv=False):
    if (deriv==True):
        return x*(1-x) #sigmoid的导数
    return 1/(1+np.exp(-x)) 
##----------------------------------------------------
## input dataset
## 每一行：样本； 每一列：输入节点/维度。  3输入节点 4个样本
#X = np.array([[0,0,1],
#              [0,1,1],
#              [1,0,1],
#              [1,1,1]])
#
## output dataset
#y = np.array([[0,0,1,1]]).T
#
## seed random numbers to make calculation 随机数生成的种子
## deterministic (just a good practice)
##设置种子的好处：
##得到的权重初始化集仍是随机分布的，但每次开始训练时，得到的权重初始集分布都是完全一致的
#np.random.seed(1)
#
## initialize weights randomly with mean 0 
## 使用零均值化来随机初始化权重
#syn0 = 2*np.random.random((3,1))-1
##syn0：零号突触：输入层-隐层的权重矩阵，
##维度（3，1）（输入维度，输出维度）l0层大小为3，l1层大小为1
##在学习训练过程中，只需存储 syn0 权值矩阵。
##所谓的“神经网络”实际上就是这个权值矩阵
#L1_error1=[]
#L1_error2=[]
#L1_error3=[]
#L1_error4=[]
#L1_delta1=[]
#L1_delta2=[]
#L1_delta3=[]
#L1_delta4=[]
## ---|开始nn训练|-------------------------------------------------
#for iter in xrange(10000): 
##循环迭代1万步，每次都用BP算法来更新权重，也是更新了1万次呀！
#    
#    # forward propagation 向前传播
#    l0 = X #第一层就是输入层，输入的原始数据
#    #4×3
#    l1 = nonlin(np.dot(l0,syn0)) #第二层即顶层输出层，
#    #输出的是sigmoid激活函数值，即预测值 
#    #nonlin函数 将一个概率值作为输出
#    #4×3 · 3×1=4×1 四个训练样本→四个预测结果
#    if iter==0: #输出一开始的预测值
#        print iter
#        print l1
#    # how much did we miss?计算误差
#    l1_error = y-l1 #误差=真值-预测值
#    L1_error1.append(-l1_error[0]) #extend(l1_error)前两个输出为0，所以应该取负号！！
#    L1_error2.append(-l1_error[1])
#    L1_error3.append(l1_error[2])
#    L1_error4.append(l1_error[3])
#    
#    # backward propagation 反向传播    
#    # multipy how much we missed by the slope of the sigmoid at the values in l1
#    # 目的：将误差输入到上一层来重新调整参数
#    l1_delta = l1_error*nonlin(l1,True) #秘密武器所在！误差项加权导数值
#    #True时nonlin输出的是导数，
#    #将“斜率”乘上误差时，实际上就在以高确信度减小预测误差。
#    L1_delta1.append(-l1_delta[0])
#    L1_delta4.append(l1_delta[3])
#    # update weights 更新权重=更新网络（网络训练中）
#    # 权值更新量=输入值*误差项加权导数值
#    syn0 +=np.dot(l0.T,l1_delta)  #网络中所有操作都是在为这步运算做准备
#    
#print "Output After Training:"
#print l1
#
#fig=plt.figure()
#ax=fig.add_subplot(411)
#plt.title('L1 Error at four output')
#x=np.arange(1,10001)#[i for i in range(1000)]
#ax.plot(x,L1_error1)
#ax=fig.add_subplot(412)
#ax.plot(x,L1_error2)
#ax=fig.add_subplot(413)
#ax.plot(x,L1_error3)
#ax=fig.add_subplot(414)
#ax.plot(x,L1_error4)
#plt.show()
#
#fig=plt.figure()
#ax=fig.add_subplot(411)
#plt.title('L1 delta at No.1 and No.4 output')
#x2=range(10000)#np.arange(1,10001)#
#ax.plot(x2,L1_delta1)
#ax=fig.add_subplot(412)
#ax.plot(x2,L1_delta4)
#plt.show()

# ====|三层神经网络|===============================================
'''
给定前两列输入（4，3），尝试去预测输出列（4，1）
非线性 模式
单个输入与输出间不存在一个一对一的关系
但输入的组合与输出间存在着一对一的关系
'''
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])
np.random.seed(1)
#0均值随机初始化权重
syn0 = 2*np.random.random((3,4))-1
syn1 = 2*np.random.random((4,1))-1

L2_error=[]#np.empty((60000,4))

for j in xrange(60000):
    # 向前传播
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1)) #l1的输出是l2的输入

    # 输出误差
    l2_error = y-12
    if j % 10000 ==0:
        print "Error="+str(np.mean(np.abs(l2_error)))
#    L2_error[0].append(l2_error[0])
#    L2_error[1].append(l2_error[1])
#    L2_error[2].append(l2_error[2])
#    L2_error[3].append(l2_error[3])
    L2_error.append(l2_error)
    # 判断目标结果的倾向
    # 如果已经非常确定的话，就不要对它大改动
        # l2的“置信度加权误差
    l2_delta = l2_error*nonlin(l2,deriv=True)
    
    # 计算l1层每个结点对来误差的贡献
    # 使用l2的“置信度加权误差”建立l1的误差
    l1_error = l2_delta.dot(syn1.T)
    
    #判断l1的输出倾向
    #如果很确定，就不要作太大改动
    #经确信度加权后的神经网络 l1 层的误差
    l1_delta = l1_error*nonlin(l1,deriv=True)
    #权重更新：    
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print "Output After Training:"
print l2
fig=plt.figure()
ax=fig.add_subplot(411)
plt.title('L2 Error at four output')
plot_x=range(60000)
plot_y1=[]
plot_y2=[]
plot_y3=[]
plot_y4=[]
for i in range(60000):
    plot_y1.append((L2_error[i][0]))
    plot_y2.append((L2_error[i][1]))
    plot_y3.append((L2_error[i][2]))
    plot_y4.append((L2_error[i][3]))
ax.plot(plot_x,plot_y1)
ax=fig.add_subplot(412)
ax.plot(plot_x,plot_y2)
ax=fig.add_subplot(413)
ax.plot(plot_x,plot_y3)
ax=fig.add_subplot(414)
ax.plot(plot_x,plot_y4)
plt.show()