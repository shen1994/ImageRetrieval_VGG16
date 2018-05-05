# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 15:59:00 2017

@author: shenxinfeng
"""

import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from PIL import Image

def data_maker(image_path):
    files_path_tmp = []
    for root, dirs, files in os.walk(image_path):
        files_path_tmp.append(dirs)
        break
    files_path = files_path_tmp[0]
   
    new_files_path = []
    for index in np.arange(0, len(files_path), 1):
        new_files_path_str = image_path + files_path[index] + '/'
        new_files_path.append(new_files_path_str)

    images_name = []
    images_label = []
    for index in np.arange(0, len(new_files_path), 1):
        n_class = index + 1
        for file_name in os.listdir(new_files_path[index]):
            if cmp(file_name[len(file_name)-  4:len(file_name)],'.jpg') == 0 :
                images_name.append(new_files_path[index] + file_name)
                images_label.append(str(n_class))
                
    images_index = np.arange(1,len(images_name) + 1,1)
    random.shuffle(images_index)
    
    image_index_col = pd.Series(images_index, name='index')
    image_name_col = pd.Series(images_name, name='name')
    image_label_col = pd.Series(images_label, name='type')
    save = pd.DataFrame({'index':image_index_col, 
                         'name':image_name_col, 
                         'type':image_label_col})
    save.to_csv(image_path + 'data.csv', index=False)
    
def_size_map = {
        'low': 224,
        'step_low': 256,
        'mid': 320,
        'step_high': 384,
        'high': 512
        }

def CreateTFRecord(read_path, out_path, is_train):
    data = pd.read_csv(read_path, header=None)    
    # 训练数据
    writer = tf.python_io.TFRecordWriter(out_path)

    if is_train == True:  
        for i in np.arange(1, data.shape[0], 1):
            img_index = long(data[0][i])
            img = Image.open(data[1][img_index])
            img_height, img_width = img.size
            img_name = data[1][img_index]
            '''
            img = img.resize((def_size_map['low'], def_size_map['low']), Image.BILINEAR) #  ANTIALIAS
            img_raw = img.tobytes()  
            example = tf.train.Example(features = tf.train.Features(feature = {
                    'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_heigt': tf.train.Feature(int64_list=tf.train.Int64List(value=[def_size_map['low']])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[def_size_map['low']])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[long(data[2][img_index])]))
                    }))
            writer.write(example.SerializeToString())
            '''
            img_raw = img.tobytes()  
            example = tf.train.Example(features = tf.train.Features(feature = {
                    'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_heigt': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_width])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[long(data[2][img_index])]))
                    }))
            writer.write(example.SerializeToString())
           
            print u'训练集:' + str(i) + u'---OK!'
    else:   
        for i in np.arange(1, data.shape[0], 1):
            img_index = long(data[0][i])
            img = Image.open(data[1][img_index])
            img_height, img_width = img.size
            img_name = data[1][img_index]
            '''
            img = img.resize((def_size_map['low'], def_size_map['low']), Image.BILINEAR) #  ANTIALIAS
            img_raw = img.tobytes()  
            example = tf.train.Example(features = tf.train.Features(feature = {
                    'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_heigt': tf.train.Feature(int64_list=tf.train.Int64List(value=[def_size_map['low']])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[def_size_map['low']])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[long(data[2][img_index])]))
                    }))
            writer.write(example.SerializeToString())
            '''
            img_raw = img.tobytes()  
            example = tf.train.Example(features = tf.train.Features(feature = {
                    'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_heigt': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_width])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[long(data[2][img_index])]))
                    }))
            writer.write(example.SerializeToString())
            
            print u'验证集:' + str(i) + u'---OK!'

    writer.close()

if __name__ == "__main__":

#=======================================================================
#========================数据类别标记
#=======================================================================

    print u'数据类别标记...'
    data_maker('./train_image/')
    data_maker('./test_image/')

#=======================================================================
#========================数据预处理
#=======================================================================

    print u'数据预处理...'
    CreateTFRecord('./train_image/data.csv', './TFRecords/train.tfrecords', is_train = True)
    CreateTFRecord('./test_image/data.csv', './TFRecords/test.tfrecords', is_train = False)

#=======================================================================
#========================数据训练VGG16
#=======================================================================
    print u'tfrecord制作完成'
    
    