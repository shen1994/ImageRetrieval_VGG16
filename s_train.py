# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 11:35:11 2017

@author: shenxinfeng
"""

import os
import shutil
import tensorflow as tf
import pandas as pd
import numpy as np

import alex_net
import vgg16
import vgg19
import inception_net_v3
import resnet
import overfeat

import data_strength

if __name__ == "__main__":
    print u'VGG16训练...'
    
    log_path = './logs'    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    shutil.rmtree(log_path)  
    os.mkdir(log_path)
    
#=================================================================================================================
    
    # 获取总样本数
    data = pd.read_csv('./train_image/data.csv', header=None)
    sample_num = data.shape[0] - 1
    test_data = pd.read_csv('./test_image/data.csv', header=None)
    test_sample_num = test_data.shape[0] - 1
    
    # 模型相关参数
    learning_rate = 5e-02 #3e-05
    train_batch = 64
    out_class = long(data[2][sample_num]) # 类别
    dropout = 0.75
    train_step = sample_num / train_batch
    decay_rate = 0.5 # 学习率衰减系数，每1000轮之后乘上0.1，精确化
    img_pll = 2 # 增加数据量， 避免过拟合
    train_equo = 24000 # 训练轮数
    train_test_keep = 1 # 多少次看一次数据
    save_module = 80000 # 多少次保存一次模型
    
#=================================================================================================================
    
    x = tf.placeholder(tf.float32, [train_batch, None, None, 3])
    y = tf.placeholder(tf.float32, [train_batch, out_class + 1])

    image_size = tf.placeholder(tf.int32)
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

#=================================================================================================================
    
    # 构建模型
    # img_vec, logits = alex_net.inference(input_op = x, keep_prob = keep_prob, output_num = out_class + 1)
    # img_vec, logits = vgg16.inference_op(input_op = x, keep_prob = keep_prob, output_num = out_class + 1)
    # img_vec, logits = vgg19.inference_op(input_op = x, keep_prob = keep_prob, output_num = out_class + 1)
    # img_vec, logits = resnet.resnet_v2_50(inputs = x, num_classes = out_class + 1)
    # img_vec, logits = inception_net_v3.inception_v3(x, num_classes = out_class + 1, keep_prob = keep_prob) #299*299
    
    img_vec, logits = overfeat.inference_op(input_op = x,
                                            image_size = image_size,
                                            keep_prob = keep_prob,
                                            output_num = out_class + 1,
                                            is_training = is_training)
 
#=================================================================================================================
    
    # train
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    #tf.add_to_collection('losses', cost)
    #l2_cost = tf.add_n(tf.get_collection('losses'), name = 'total_loss')
    global_step = tf.Variable(0, trainable = False)
    lr = tf.train.exponential_decay(learning_rate, global_step, 2000, decay_rate, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost, global_step=global_step)    
    pred = tf.nn.softmax(logits)
    pred = tf.reshape(pred,[train_batch, out_class + 1])
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # test
    ''' 
    out_sum = tf.zeros([train_batch, out_class + 1], dtype=tf.float32) 
    last_conv_softmax_0 = tf.nn.softmax(last_conv_0)
    last_conv_softmax_1 = tf.nn.softmax(last_conv_1)
    last_conv_softmax_2 = tf.nn.softmax(last_conv_2)
    last_conv_softmax_3 = tf.nn.softmax(last_conv_3)
    
    out_sum = tf.add(out_sum, last_conv_softmax_0)
    out_sum = tf.add(out_sum, last_conv_softmax_1)
    out_sum = tf.add(out_sum, last_conv_softmax_2)
    out_sum = tf.add(out_sum, last_conv_softmax_3)
    out_cons = tf.to_float(tf.fill([train_batch, out_class + 1], 4.0))
    out_mean = tf.div(out_sum, out_cons)
    out_mean = tf.reshape(out_mean,[train_batch, out_class + 1])
    
    t_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_mean, labels=t_y))    
    t_correct_pred = tf.equal(tf.argmax(out_mean, 1), tf.argmax(t_y, 1))
    t_accuracy = tf.reduce_mean(tf.cast(t_correct_pred, tf.float32)) 
    ''' 
    '''
    last_conv_softmax_0 = tf.nn.softmax(logits_0)
    last_conv_softmax_1 = tf.nn.softmax(logits_1)
    last_conv_softmax_2 = tf.nn.softmax(logits_2)
    last_conv_softmax_3 = tf.nn.softmax(logits_3)
    
    last_conv_max_0 = tf.nn.top_k(last_conv_softmax_0, 1)
    last_conv_max_1 = tf.nn.top_k(last_conv_softmax_1, 1)
    last_conv_max_2 = tf.nn.top_k(last_conv_softmax_2, 1)
    last_conv_max_3 = tf.nn.top_k(last_conv_softmax_3, 1)
    '''        
    #last_conv_softmax_out = tf.cond(last_conv_max_0 < last_conv_max_1, lambda: last_conv_softmax_1, lambda: last_conv_softmax_0)
    #last_conv_max_out = tf.nn.top_k(last_conv_softmax_out, 1)

#=================================================================================================================
    
    saver = tf.train.Saver()
    '''    
    tf.add_to_collection('vec_0', img_vec_0)
    tf.add_to_collection('logits_0', logits_0) # 保存输出格式
    tf.add_to_collection('vec_1', img_vec_1)
    tf.add_to_collection('logits_1', logits_1) # 保存输出格式
    tf.add_to_collection('vec_2', img_vec_2)
    tf.add_to_collection('logits_2', logits_2) # 保存输出格式
    tf.add_to_collection('vec_3', img_vec_3)
    tf.add_to_collection('logits_3', logits_3) # 保存输出格式
    '''
    # TFRecord.important_test()

#=================================================================================================================
    
    # 获取数据
    
    # 224*224 random cut
    images, ori_images, labels = data_strength.read_and_decode(file_name = './TFRecords/train.tfrecords',
                                                output_num = out_class + 1,
                                                img_size = 224,
                                                img_num = img_pll)
    
    train_feed_data = [] # cut image
    train_ori_feed_data = [] # ori image
    
    image_batch, ori_image_batch, label_batch = tf.train.shuffle_batch([images, ori_images, labels], 
                                                        batch_size = train_batch,
                                                        num_threads = 16,
                                                        capacity = int(0.4 * sample_num) + 3 * train_batch,
                                                        min_after_dequeue = int(0.4 * sample_num))
    
    for i in range(2):
        img_tmp = tf.slice(ori_image_batch, [0, i, 0, 0, 0], [train_batch, 1, 224, 224, 3])
        img_tmp = tf.reshape(img_tmp, [train_batch, 224, 224, 3])
        train_ori_feed_data += [img_tmp, label_batch]
        
    for i in range(img_pll + 1):
        img_tmp = tf.slice(image_batch, [0, i, 0, 0, 0], [train_batch, 1, 224, 224, 3])
        img_tmp = tf.reshape(img_tmp, [train_batch, 224, 224, 3])
        train_feed_data += [img_tmp, label_batch]
        
    #224 * 224 256 * 256 320 * 320 384 * 384
    t_images_0, t_images_1, t_images_2, t_images_3, t_labels = data_strength.test_read_and_decode(file_name = './TFRecords/test.tfrecords',
                                                                 output_num = out_class + 1)
    
    t_images_batch_0, t_images_batch_1, t_images_batch_2, t_images_batch_3, t_labels_batch = \
            tf.train.shuffle_batch([t_images_0, t_images_1, t_images_2, t_images_3, t_labels],
                                    batch_size = train_batch, 
                                    num_threads = 16,
                                    capacity = int(0.4 * test_sample_num) + 3 * train_batch,
                                    min_after_dequeue = int(0.4 * test_sample_num)
                                    )
     
#=================================================================================================================
            
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('acc', accuracy)
    #tf.summary.scalar('t_loss', t_cost)
    #tf.summary.scalar('t_acc', t_accuracy)
    merged = tf.summary.merge_all()
   
    with tf.Session() as sess:       
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
         
        
        train_summary_writer = tf.summary.FileWriter('logs/train', sess.graph) 
        test_summary_writer = tf.summary.FileWriter('logs/test', sess.graph)
        
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        
        step = 1
        while step < train_equo:
            # 原数据---增强
            print 's'
            for i in range(2):
                _image, _label = sess.run([train_ori_feed_data[i * 2], train_ori_feed_data[i * 2 + 1]])
                a,b=sess.run([img_vec, logits], feed_dict={x: _image, image_size: 224, keep_prob: 1., is_training: True})
               # sess.run(optimizer, feed_dict={x: _image, y: _label, image_size: 224, keep_prob: dropout, is_training: True})   
            # 扩增数据---增强
            print 'c'
            for i in range(img_pll + 1):
                _image, _label = sess.run([train_feed_data[i * 2], train_feed_data[i * 2 + 1]])              
                sess.run(optimizer, feed_dict={x: _image, y: _label, image_size: 224, keep_prob: dropout, is_training: True})
            print 'v'
            if step % train_test_keep == 0:
                _image, _label = sess.run([train_ori_feed_data[0], train_ori_feed_data[1]]) 
                
                train_summary = sess.run(merged, feed_dict={ \
                        x: _image, y: _label, image_size: 224, keep_prob: 1., is_training: True})
                train_summary_writer.add_summary(train_summary, step)
                train_summary_writer.flush()
                
                '''   
                _image, _label = sess.run([train_ori_feed_data[0], train_ori_feed_data[1]])
                _t_images_0, _t_images_1, _t_images_2, _t_images_3, _t_labels = \
                        sess.run([t_images_batch_0, t_images_batch_1, t_images_batch_2, t_images_batch_3, t_labels_batch])
                train_summary = sess.run(merged, feed_dict={ \
                        x: _image, y: _label, keep_prob: 1.,
                        x_0: _t_images_0, x_1: _t_images_1, x_2: _t_images_2,  x_3: _t_images_3, y: _t_labels, keep_prob: 1.})
                train_summary_writer.add_summary(train_summary, step)
                train_summary_writer.flush()
                '''
                '''
                _t_images_0, _t_images_1, _t_images_2, _t_images_3, _t_labels = \
                           sess.run([t_images_batch_0, t_images_batch_1, t_images_batch_2, t_images_batch_3, t_labels_batch])
                
                # 窗口求和
                s_0, s_1, s_2, s_3 = sess.run([last_conv_softmax_0, last_conv_softmax_1, last_conv_softmax_2, last_conv_softmax_3], feed_dict={                        
                        x_0: _t_images_0, x_1: _t_images_1, x_2: _t_images_2,  x_3: _t_images_3, y: _t_labels, keep_prob: 1.})
                s_all = (s_0 + s_1 + s_2 + s_3) / 4.0
                
                indices = []
                t_indices = []
                
                for j in range(train_batch):
                    indices += [np.argmax(s_all[j])]
                    t_indices += [np.argmax(_t_labels[j])]
                    
                t_count = 0                    
                for j in range(train_batch):
                    if(indices[j] == t_indices[j]):
                        t_count += 1
                        
                t_sum_acc = float(t_count) / float(train_batch)
                
                # 窗口求最大            
                t_0, t_1, t_2, t_3 = sess.run([last_conv_max_0, last_conv_max_1, last_conv_max_2, last_conv_max_3], feed_dict={                        
                        x_0: _t_images_0, x_1: _t_images_1, x_2: _t_images_2,  x_3: _t_images_3, y: _t_labels, keep_prob: 1.}) 

                pro = []
                indices = []

                for j in range(train_batch):
                    if (t_0.values[j] - t_1.values[j]) > 0:
                        pro += [t_0.values[j]]
                        indices += [t_0.indices[j]]
                    else:
                        pro += [t_1.values[j]]
                        indices += [t_1.indices[j]]
                        
                    if (pro[j] - t_2.values[j]) > 0:
                        pass
                    else:
                        pro[j] = [t_2.values[j]]
                        indices[j] = [t_2.indices[j]]
                        
                    if (pro[j] - t_3.values[j]) > 0:
                        pass
                    else:
                        pro[j] = [t_3.values[j]]
                        indices[j] = [t_3.indices[j]]
                
                t_indices = []
                for j in range(train_batch): 
                    t_indices += [np.argmax(_t_labels[j])]

                t_count = 0               
                for j in range(train_batch):
                    if indices[j] == t_indices[j]:
                        t_count += 1
                        
                t_diff_acc = float(t_count) / float(train_batch)
                
                # 窗口求原图
                t_count = 0
                for j in range(train_batch):
                    if t_0.indices[j][0] == t_indices[j]:
                        t_count += 1
                        
                t_acc = float(t_count) / float(train_batch)

                print step, t_diff_acc, t_sum_acc, t_acc
            
            if step % save_module == 0:
            	saver.save(sess, './train_module/classify_model.ckpt', \
            	           global_step = step / 1000 * train_batch, latest_filename=None,
            	           meta_graph_suffix='meta', write_meta_graph=True, write_state=True)
            '''
            step += 1
            print step
         
        coord.request_stop()
        coord.join(threads)
        
    train_summary_writer.close()
    test_summary_writer.close()
    sess.close()

    print u'程序结束...'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    