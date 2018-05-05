# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 17:24:18 2017

@author: shenxinfeng
"""

import tensorflow as tf

def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image,max_delta=32./255.)#亮度
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)#饱和度
        image = tf.image.random_hue(image,max_delta=0.2)#色相
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)#对比度
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
    return tf.clip_by_value(image,0.0,1.0) #将张量值剪切到指定的最小值和最大值

# 参考cifar-10 distorted_inputs
def preprocess_for_train(image, height, width, img_num):
    
    # multi_scale
    img_batch = []
    img_ori_batch = []
    smin = 256
    smax = 384
    scale_rate = (smax - smin) / img_num
    for i in range(img_num + 1):
        image_size = int(smin + scale_rate * i)
        
        distort_image = tf.image.resize_images(image, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
        
        distort_image = tf.cast(distort_image, tf.float32) 
    
        # 剪裁或是改变原图大小输入
        distort_image = tf.random_crop(distort_image, [height, width, 3])
        # distort_image = tf.resize_image_with_crop_or_pad(image, height, width)
        # distort_image = tf.image.resize_images(image, [height, width], method=0)
    
        #随机左右翻转图像
        distort_image = tf.image.random_flip_left_right(distort_image)
        
        #使用一种随机的顺序调整图像色彩
        # distort_image = distort_color(distort_image,np.random.randint(1))
         
        distort_image = tf.image.random_brightness(distort_image,  
                                                   max_delta=63)  
        distort_image = tf.image.random_contrast(distort_image,  
                                                 lower=0.2, upper=1.8) 
        
        # Subtract off the mean and divide by the variance of the pixels.
        # tf.image.per_image_whitening has been replaced by tf.image.per_image_standardization
        float_image = tf.image.per_image_standardization(distort_image)
      
        float_image.set_shape([height, width, 3]) 
        
        img_batch += [float_image]
     
    # ori_img
    distort_image = tf.image.resize_images(image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    distort_image = tf.cast(distort_image, tf.float32) 
    float_image = tf.image.per_image_standardization(distort_image)   
    float_image.set_shape([height, width, 3]) 
    img_ori_batch += [float_image]
    
    # ori_random_img   
    distort_image = tf.image.resize_images(image, [height, width], method=tf.image.ResizeMethod.BILINEAR) 
    distort_image = tf.cast(distort_image, tf.float32)
    distort_image = tf.image.random_flip_left_right(distort_image)
    distort_image = tf.image.random_brightness(distort_image,  
                                               max_delta=63)  
    distort_image = tf.image.random_contrast(distort_image,  
                                               lower=0.2, upper=1.8) 
    float_image = tf.image.per_image_standardization(distort_image) 
    float_image.set_shape([height, width, 3]) 
    img_ori_batch += [float_image]
    
    return img_batch, img_ori_batch


# 参考cifar-10 inputs
def preprocess_for_test(image, height, width):
    
    img_batch = []
    img_ori_batch = []
    
    # oringe image
    ori_image = tf.image.resize_images(image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    ori_image = tf.cast(ori_image, tf.float32)
    
    for i in range(2):
        if i == 0:
            pass
        else:
            ori_image = tf.image.flip_left_right(ori_image)
 
        float_image = tf.image.per_image_standardization(ori_image)
        float_image.set_shape([height, width, 3])
        
        img_ori_batch += [float_image]

    # crop image
    crop_image = tf.image.resize_images(image, [384, 384], method=tf.image.ResizeMethod.BILINEAR) # 256 > 224
    crop_image = tf.cast(crop_image, tf.float32)
    
    for i in range(2):
        if i == 0:
            pass
        else:
            crop_image = tf.image.flip_left_right(crop_image)

        # 中心
        float_image0 = tf.slice(crop_image, [80, 80, 0], [height, width, 3]) # (384 - 224) / 2 = 80
        float_image0 = tf.image.per_image_standardization(float_image0)
        float_image0.set_shape([height, width, 3])
             
        img_batch += [float_image0]
        
        # 左上
        float_image1 = tf.slice(crop_image, [0, 0, 0], [height, width, 3])
        float_image1 = tf.image.per_image_standardization(float_image1)
        float_image1.set_shape([height, width, 3]) 
        
        img_batch += [float_image1]
    
        # 右上
        float_image2 = tf.slice(crop_image, [0, 160, 0], [height, width, 3]) # 384 - 224 = 160
        float_image2 = tf.image.per_image_standardization(float_image2)
        float_image2.set_shape([height, width, 3])  
        
        img_batch += [float_image2]
    
        # 左下
        float_image3 = tf.slice(crop_image, [160, 0, 0], [height, width, 3]) # 384 - 224 = 160
        float_image3 = tf.image.per_image_standardization(float_image3)
        float_image3.set_shape([height, width, 3])  
        
        img_batch += [float_image3]
    
        # 右下
        float_image4 = tf.slice(crop_image, [160, 160, 0], [height, width, 3]) # 384 - 224 = 160
        float_image4 = tf.image.per_image_standardization(float_image4)
        float_image4.set_shape([height, width, 3])  
        
        img_batch += [float_image4]

    
    return img_batch, img_ori_batch

def_size = 224

def read_and_decode(file_name,output_num, img_num):
    filename_queue = tf.train.string_input_producer([file_name], shuffle=False, num_epochs = None)  

    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)   
    
    features = tf.parse_single_example(serialized_example, features={ 
                'img_raw': tf.FixedLenFeature([], tf.string),
                'img_heigt':tf.FixedLenFeature([], tf.int64),
                'img_width':tf.FixedLenFeature([], tf.int64),
                'label':tf.FixedLenFeature([], tf.int64)
                })
  
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img_heigt = tf.cast(features['img_heigt'], tf.int32) 
    img_width = tf.cast(features['img_width'], tf.int32) 
    img = tf.reshape(img,[img_heigt, img_width, 3]) 

    img_batch, ori_img_batch = preprocess_for_train(img, def_size, def_size, img_num)
    
    # 如果labels的每一行是one-hot表示，也就是只有一个地方为1，其他地方为0
    # 可以使用tf.sparse_softmax_cross_entropy_with_logits()
    label = tf.cast(features['label'], tf.int32)  
    label = tf.one_hot(indices = label, depth = output_num, 
                       on_value = 1., off_value = 0., axis = -1)

    return img_batch, ori_img_batch, label

def test_read_and_decode(file_name,output_num):
    filename_queue = tf.train.string_input_producer([file_name], shuffle=False, num_epochs = None)  

    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)   
    
    features = tf.parse_single_example(serialized_example, features={ 
                'img_raw': tf.FixedLenFeature([], tf.string),
                'img_heigt':tf.FixedLenFeature([], tf.int64),
                'img_width':tf.FixedLenFeature([], tf.int64),
                'label':tf.FixedLenFeature([], tf.int64)
                })  
  
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img_heigt = tf.cast(features['img_heigt'], tf.int32) 
    img_width = tf.cast(features['img_width'], tf.int32) 
    img = tf.reshape(img,[img_heigt, img_width, 3]) 
    
    img_batch, ori_img_batch = preprocess_for_test(img, def_size, def_size)
    
    label = tf.cast(features['label'], tf.int32)  
    label = tf.one_hot(indices = label, depth = output_num, 
                       on_value = 1., off_value = 0., axis = -1)

    return img_batch, ori_img_batch, label

def important_test():

    count = 0
    
    sess = tf.Session()
    example = tf.train.Example()
    
    for serialized_example in tf.python_io.tf_record_iterator('./TFRecords/train.tfrecords'):
        count += 1
        
        if count > 0:
            
            example.ParseFromString(serialized_example)
        
            img = example.features.feature['img_raw'].bytes_list.value[0]
            img = tf.decode_raw(img, tf.uint8)
            img_heigt = example.features.feature['img_heigt'].int64_list.value[0]
            img_heigt = tf.cast(img_heigt, tf.int32)
            img_width = example.features.feature['img_width'].int64_list.value[0]
            img_width = tf.cast(img_width, tf.int32)
            img = tf.reshape(img,[img_heigt, img_width, 3]) 
            img = tf.image.per_image_standardization(img)
            
            label = example.features.feature['label'].int64_list.value[0]
            label = tf.cast(label, tf.int32)  
            label = tf.one_hot(indices = label, depth = 14, 
                           on_value = 1., off_value = 0., axis = -1)
            
            
            
            img_name = example.features.feature['img_name'].bytes_list.value[0]
            
            print str(count) + '---' + img_name
            
            _ = sess.run(img)
        
    sess.close




