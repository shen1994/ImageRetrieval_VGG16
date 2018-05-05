# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 14:50:26 2017

@author: shenxinfeng
"""

import os
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

if __name__ == "__main__": 

#=======================================================================
#========================数据类别标记
#=======================================================================

    print u'创建索引表...'
    
    image_path = './image/'
    files_path_tmp = []
    for root, dirs, files in os.walk(image_path):
        files_path_tmp.append(dirs)
        break
    files_path = files_path_tmp[0]
   
    new_files_path = []
    for index in np.arange(0, len(files_path), 1):
        new_files_path_str = image_path + files_path[index] + '/'
        new_files_path.append(new_files_path_str)

    images_path = []
    images_label = []
    sample_num = 0
    vec_type = []
    vec_name = []
    vec_start = []
    vec_end = []
    for index in np.arange(0, len(new_files_path), 1):
        n_class = index + 1
        vec_type.append(str(n_class))
        vec_name.append(new_files_path[index] + str(n_class) + '_vec_txt.txt')
        vec_start.append(str(sample_num + 1))
        for file_name in os.listdir(new_files_path[index]):
            if cmp(file_name[len(file_name)-  4:len(file_name)],'.jpg') == 0 :
                sample_num += 1
                images_path.append(new_files_path[index] + file_name)
                images_label.append(str(n_class))
        vec_end.append(str(sample_num))
    
    image_path_col = pd.Series(images_path, name='path')
    image_label_col = pd.Series(images_label, name='type')
    save = pd.DataFrame({'path':image_path_col,
                         'type':image_label_col})
    save.to_csv('./image/data.csv', index=False)
    
    vec_name_col = pd.Series(vec_name, name='name')
    vec_type_col = pd.Series(vec_type, name='type')
    vec_start_col = pd.Series(vec_start, name='start')
    vec_end_col = pd.Series(vec_end, name='end')
    vec_save = pd.DataFrame({'name':vec_name_col, 
                             'type':vec_type_col, 
                             'start':vec_start_col, 
                             'end':vec_end_col})
    vec_save.to_csv('./image/vec_data.csv', index=False)

#=======================================================================
#========================得到特征向量
#=======================================================================
    
    print u'创建相似度表...'

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
    
        path = './image/data.csv'
        vec_path = './image/vec_data.csv'
        data = pd.read_csv(path, header=None)
        vec_data = pd.read_csv(vec_path, header=None)
        
        all_img_vec = []
        for i in np.arange(1, vec_data.shape[0], 1):
            for j in np.arange(long(vec_data[2][i]), long(vec_data[0][i]) + 1, 1):
                img = cv2.imread(data[0][j])
                pp = cv2.resize(img, (224, 224))
                pp = np.asarray(pp, dtype=np.float32)
                pp /= 255.
                pp = pp.reshape((pp.shape[0], pp.shape[1], 3))
                
                img_vec, pred = sess.run([this_img_vec, this_pred], 
                                         feed_dict={this_x:[pp], this_keep_prob: 1.})
                img_vec = img_vec[0]
                all_img_vec.append(img_vec)
                
            print u'特征准备:' + str(i) + '---OK!'
    
    sess.close()  

#=======================================================================
#========================PCA降维
#=======================================================================   
    
    print u'PCA降维...'
    
    K = 256
    
    pca = PCA(n_components=256)
    pca_img_vec = pca.fit_transform(all_img_vec)
    np.savetxt('./image/vec.txt', pca_img_vec)
    
    vec_cov = np.dot(np.transpose(all_img_vec), all_img_vec)
    vec_cov = vec_cov / len(vec_cov)
    
    vec_u, vec_sigma, vec_v = np.linalg.svd(vec_cov)
    
    UK = []
    m = len(vec_u)
    for k in np.arange(K):
        uk = vec_u[:, k].reshape(m, 1)
        UK.append(uk)

    np.savetxt('./image/feature_vec.txt', UK)    
    
#=======================================================================
#========================创建结束
#======================================================================= 
    print u'创建结束...'
    
    
        
    
    