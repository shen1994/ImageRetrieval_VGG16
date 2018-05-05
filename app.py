# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:10:16 2017

@author: shenxinfeng
"""

import os
import cv2
import shutil
import tensorflow as tf
import pandas as pd
import numpy as np

import vector_distance

from PIL import Image

if __name__ == "__main__":
    check_image_path = './test1.jpg'
#=======================================================================
#========================取出一张图片得到分类结果
#=======================================================================
    
    print u'获取类别和特征提取...'
        
    img = cv2.imread(check_image_path)
    pp = cv2.resize(img, (224, 224))
    pp = np.asarray(pp, dtype=np.float32)
    pp /= 255.
    pp = pp.reshape((pp.shape[0], pp.shape[1], 3))

    ckpt = tf.train.latest_checkpoint('./train_module') 

    saver = tf.train.import_meta_graph(ckpt + '.meta')

    this_x = tf.get_collection("x")[0]
    this_pred = tf.get_collection("pred")[0]
    this_keep_prob = tf.get_collection("keep_prob")[0]
    this_img_vec = tf.get_collection("vec")[0]

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
      
        saver.restore(sess, ckpt) 
        
        img_vec, pred = sess.run([this_img_vec, this_pred], feed_dict={this_x:[pp], this_keep_prob: 1.})
        img_vec = img_vec[0]
        
        _class =  sess.run(tf.argmax(pred ,1))
        
        print u'类别：' + str(_class)

    sess.close()
    
    _class_vec = []
    vec_index = np.argsort(pred[0])
    for i in np.arange(3):
        _class_vec.append(vec_index[len(vec_index) - i - 1])
        
    print u'模糊的类别：' + str(_class_vec)
    
#=======================================================================
#========================图片整体降维
#=======================================================================
    
    print u'图片降维...'
      
    UK = np.loadtxt('./image/feature_vec.txt')
    
    pca_img_vec = np.dot(UK, img_vec)

#=======================================================================
#========================图片匹配
#=======================================================================
    
    print u'图片匹配...'
    
    img_size = 10
    
    dis_pro = [0 for i in np.arange(3)]
    dis_pro[0] =  0.35
    dis_pro[1] =  0.35
    dis_pro[2] =  0.30

    vec_path = './image/vec_data.csv'
    vec_data = pd.read_csv(vec_path, header=None)
    vec_lines = np.loadtxt('./image/vec.txt')
    
    vec_len = 0
    vec_img = []
    vec_dis = []
    
    for i in np.arange(3):
        start = long(vec_data[2][_class_vec[i]])
        end = long(vec_data[0][_class_vec[i]])
        for j in np.arange(start, end + 1, 1):
            _vec = vec_lines[j - 1]
            vec_img.append(j)
            vec_dis.append(dis_pro[i] * vector_distance.manhattan_distance(pca_img_vec, _vec))
            vec_len += 1
    
    vec_index = np.argsort(vec_dis)
    
    similar_path = './similar_image'    
    if not os.path.exists(similar_path):
        os.makedirs(similar_path)
    shutil.rmtree(similar_path)  
    os.mkdir(similar_path)
    
    path = './image/data.csv'
    data = pd.read_csv(path, header=None)

    for i in np.arange(img_size):
        index = vec_index[vec_len - i - 1]
        save_img = Image.open(data[0][vec_img[index]])
        save_img.save(similar_path + '/' + str(i + 1) + '.jpg')
        
    print u'查找结束...'  

#=======================================================================
#========================至此结束
#=======================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    