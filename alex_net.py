# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:36:49 2017

@author: shen1994
"""

# AlexNet包含6亿3000万个连接，6000万个参数和65万个神经元，拥有5个卷积层
# 其中3个卷积层后面连接了最大池化层，最后还有3个全连接层
# 优势：
# 1. 使用ReLU作为CNN的激活函数，并验证其效果在较深的网络中超过了Sigmoid
# 成功解决了Sigmoid在网络较深时的梯度弥散问题
# 2. 训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合
# 主要在最后几个全连接层使用了Dropout
# 3. 在CNN中使用重叠的最大池化。此前CNN中普遍使用平均池化，产生了模糊化效果
# 在AlexNet中提出的步长比池化核的尺寸要小
# 4. 提出LRN层，对局部神经元的活动创建竞争机制
# 使其中响应比较大的值变得相对比较大，并抑制其他反馈较小的神经元
# 5. 使用CUDA加速深度卷积网络的训练，利用GPU强大的并行计算能力，处理神经网络训练时大量的矩阵运算
# AlexNet使用了两块GTX580 GPU进行训练，单个GTX580只有3GB显存
# 每个GPU桑存储一半的神经元参数，GPU之间通信非常方便
# 6. 数据增强，随机从256*256的原始图像中截取224*224大小的区域（以及水平翻转的镜像）
# 相当于增加了（256-224)^2*2=2048倍的数据量，减轻过拟合，提高泛化能力
# 对图像的RGB数据进行PCA处理，并对主成分做一个标准差为0.1的高斯扰动
# 增加一些噪声，这个Trick可以让错误率再下降1%

import tensorflow as tf

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

def inference(images, keep_prob, output_num):
    p = []
    # tf.name_scope('conv1') as scope可以将scope内部生成的Variable自动命名为conv1/xxx
    # 便于区分不同卷积层之间的组件
    # 使用print_activations将这一层最后输出的tensor conv1的结构打印出来
    # 将这一层可训练的参数kernel、biases添加到parameters中
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        p += [kernel, biases]
        
    # 在第一个卷积层后再添加LRN层和最大池化层
    # depth_radius设置为4，1.0，0.001/9.0，0.75都是AlexNet论文中推荐值
    # 除了Alexnet，其他经典的卷积神经网络模型放弃了LRN（效果不明显），前馈和反馈的速度降到1/3
    with tf.name_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

        # pool1
        pool1 = tf.nn.max_pool(lrn1,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1],
                              padding='VALID',
                              name='pool1')

    # 卷积核尺寸是5*5，卷积步长设置为1，即扫描全图像像素
    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        p += [kernel, biases]
          
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)
              
        # pool2
        pool2 = tf.nn.max_pool(lrn2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool2')

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        p += [kernel, biases]

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        p += [kernel, biases]

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        p += [kernel, biases]

    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
  
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
  
 