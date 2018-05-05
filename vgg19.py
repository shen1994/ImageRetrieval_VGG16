# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:42:47 2017

@author: shenxinfeng
"""

import tensorflow as tf

# 定义卷积层的函数
# input_op是输入的tensor,name是这一层的名字
# kh是kernel_height即卷积和的高，kw是kernel_width即卷积核的宽
# n_out是卷积核的数量即输出通道数
# dh是步长的高，dw是步长的宽，p是参数列表
# get_shape()[-1].value获取输入input_op的通道数
# 比如输入图片的尺寸224*224*3，则获取的数为3
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation
    
# 定义全连接的函数
# 这里的biases不再初始化为0，而是赋予一个较小的值0.1，以避免dead neuron
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias_init_val = tf.constant(0.1, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation
    
# 定义最大池化层
def max_pool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)
    
# VGGNet-16只要分为6个部分，前5段为卷积网络，最后一段是全连接网络
def inference_op(input_op, keep_prob, output_num):
    p = []
    
    # 第一段卷积网络由2个卷积网络和1个最大池化层构成
    # 卷积核大小为3*3，卷积核数量为64，步长为1*1，全像素扫描
    # 第一个卷积层的输入224*224*3，输出为224*224*3  560*560*3
    # 第二个卷积层的输入224*224*64，输出为224*224*64 560*560*64
    # 第三个池化层为标准的2*2最大池化，输出尺寸为112*112*64 280*280*64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = max_pool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)
    
    # 第二段卷积网络由2个卷积网络和1个最大池化层构成
    # 两个卷积层的卷积核是3*3，输出通道数 变为128 280*280*128
    # 最大池化层保持一致，所以输出尺寸为56*56*128 140*140*128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = max_pool_op(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)
    
    # 第三段卷积网络由3个卷积网络和1个最大池化层构成
    # 三个卷积层的卷积核是3*3，输出通道数 变为256 140*140*256
    # 最大池化层保持一致，所以输出尺寸为28*28*256 70*70*256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_4 = conv_op(conv3_3, name="conv3_4", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = max_pool_op(conv3_4, name="pool3", kh=2, kw=2, dw=2, dh=2)
    
    # 第四段卷积网络由3个卷积网络和1个最大池化层构成
    # 三个卷积层的卷积核是3*3，输出通道数 变为512 70*70*512
    # 最大池化层保持一致，所以输出尺寸为14*14*512 35*35*512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_4 = conv_op(conv4_3, name="conv4_4", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = max_pool_op(conv4_4, name="pool4", kh=2, kw=2, dw=2, dh=2)
    
    # 第5段卷积输出通道数不变512
    # 池化层输出的尺寸变为7*7*512 7*7*512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_4 = conv_op(conv5_3, name="conv5_4", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = max_pool_op(conv5_4, name="pool5", kh=2, kw=2, dw=2, dh=2)
    
    # 将第5段卷积网络扁平化，使用tf.reshape将每个样本长度转化为7*7*512=25088的一维向量
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")
    
    # 创建第一个全连接层
    # 连接一个隐含节点数为4096的全连接层，激活函数为ReLU，然后连接一个Dropou层
    # 在训练节点时保留率0.5，预测时为1.0
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
    
    # 创建第二个全连接层
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
    
    # 创建第三个连接层
    fc8 = fc_op(fc7_drop, name="fc8", n_out = output_num, p=p)
    
    return fc7_drop, fc8
      