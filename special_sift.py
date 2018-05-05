# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:23:01 2017

@author: shenxinfeng
"""

import os
import pylab as pb
import pandas as pd
from PIL import Image
import numpy as np
import shutil
import cv2

def image_to_des(image_path):
    img = Image.open(image_path).convert('L') #图像灰度化
    new_img = img.resize((560, 560), Image.BILINEAR) #ANTIALIAS,BILINEAR
        
    exe_path = np.array(os.getcwd())
    for i in np.arange(len(exe_path.data)):
        if cmp(exe_path.data[i],"\\") == 0:
            exe_path.data[i] = '/'
    
    # print '第一张图片-提取描述子...'
    new_img.save('./tmp/img_tmp.pgm')
    img_name = "./tmp/img_tmp.pgm"
    cmmd = str(exe_path.data + "/tools/sift.exe " + img_name \
                + " --output=" + "./tmp/sift.txt" + " --edge-thresh 10 --peak-thresh 5")
    # os.system(cmmd1) 显示运行终端
    os.popen(cmmd)
    sift_file = pb.loadtxt('./tmp/sift.txt')

    # kp1 = sift_file1[:,:4] #特征位置
    des = sift_file[:,4:] #描述子
    
    # print '开始匹配...'
    des = np.array([d/np.linalg.norm(d) for d in des]) #描述子标准化，去除光照影响
    
    d_des = des.shape[0]
    size_des = len(des.data) / des.shape[0]
    
    return d_des, size_des, des

def linux_image_to_des(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray_img, None)
    
    des = np.array([d/np.linalg.norm(d) for d in des]) #描述子标准化，去除光照影响
    d_des = des.shape[0]
    size_des = len(des.data) / des.shape[0]
    
    return d_des, size_des, des

def save_sift_des():
#=======================================================================
#========================创建目录
#=======================================================================
    description_path = './image_description'
    if not os.path.exists(description_path):
        os.makedirs(description_path)
    shutil.rmtree(description_path)  
    os.mkdir(description_path)
    
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
        
    for index in np.arange(0, len(new_files_path), 1):
        file_file_name = new_files_path[index][len(image_path):len(new_files_path[index])-1]
        if not os.path.exists(description_path + '/' + file_file_name):
            os.makedirs(description_path + '/' + file_file_name)
        shutil.rmtree(description_path + '/' + file_file_name)  
        os.mkdir(description_path + '/' + file_file_name)
        
    print u'完成描述子存储...'
        
#=======================================================================
#========================创建sift描述子文件
#=======================================================================
    txt_name = []
    txt_size = []
    for index in np.arange(0, len(new_files_path), 1):
        for file_name in os.listdir(new_files_path[index]):
            if cmp(file_name[len(file_name)-  4:len(file_name)],'.jpg') == 0 :
                #print file_name                
                d_sift_des, size_sift_des, sift_des = linux_image_to_des(new_files_path[index] + file_name)
                # 存储sift描述子
                sift_path = description_path + '/' + \
                new_files_path[index][len(image_path):len(new_files_path[index])-1] + \
                '/' + file_name[0:len(file_name)-  5] + '.txt'
                np.savetxt(sift_path,sift_des)
                # 存储表格
                txt_name.append(sift_path)
                txt_size.append(str(d_sift_des))
                print new_files_path[index] + file_name
    pd.Series(txt_name, name='name')
    pd.Series(txt_size, name='size')
    save = pd.DataFrame({'name':txt_name, 'size':txt_size})
    save.to_csv('./image_description/des_data.csv', index=False)              
#=======================================================================
#========================创建表格
#=======================================================================
    
def sift_macth_num(des1, des2):
    # 一般取值0.4-0.6
    # 0.4 对于准确度要求高的匹配
    # 0.5 一般情况下
    # 0.6 对于匹配点数目要求比较多的匹配
    ratio = 0.8

    # print '第一张图在第二张图中的匹配...'
    #match_scores_1 = np.zeros((des1.shape[0], 1), 'int') #176个描述子
    des2_t = des2.T
    match_count = 0
    for i in np.arange(des1.shape[0]):
        dotprobs_1 = np.dot(des1[i, :], des2_t)
        dotprobs_1 = 0.999999 * dotprobs_1
        index_1 = np.argsort(np.arccos(dotprobs_1))
        #检查最近邻的角度是否小于dist_ratio乘以第二近邻的角度
        if np.arccos(dotprobs_1)[index_1[0]] < ratio * np.arccos(dotprobs_1)[index_1[1]]:
            #match_scores_1[i] = int(index_1[0])
            match_count += 1
            
    '''
    # print '第二张图在第一张图中的匹配...'        
    match_scores_2 = np.zeros((des2.shape[0], 1), 'int') #669个描述子
    des1_t = des1.T
    for i in np.arange(des2.shape[0]):
        dotprobs_2 = np.dot(des2[i, :], des1_t)
        dotprobs_2 = 0.999999 * dotprobs_2
        index_2 = np.argsort(np.arccos(dotprobs_2))
        #检查最近邻的角度是否小于dist_ratio乘以第二近邻的角度
        if np.arccos(dotprobs_2)[index_2[0]] < ratio * np.arccos(dotprobs_2)[index_2[1]]:
            match_scores_2[i] = int(index_2[0])
    
    # print '去除不对称的匹配...'
    ndx_12 = match_scores_1.nonzero()[0]
    for n in ndx_12:
        if match_scores_2[int(match_scores_1[n])] != n:
            match_scores_1[n] = 0
        
    #print match_scores_1

    match_count = 0
    #print match_scores_1.shape[0]
    for i in np.arange(match_scores_1.shape[0]):
        if match_scores_1[i][0] != 0:
            match_count += 1    
    '''
    
    return match_count
    

def sift_match(check_image_path, image_class, match_rate):

    des_path = './image_description/'
    files_path_tmp = []
    for root, dirs, files in os.walk(des_path):
        files_path_tmp.append(dirs)
        break
    files_path = files_path_tmp[0]
   
    new_files_path = []
    for index in np.arange(0, len(files_path), 1):
        new_files_path_str = des_path + files_path[index] + '/'
        new_files_path.append(new_files_path_str)
    
    match_file_names = []
    d_des1, size_des1, des1 = image_to_des(check_image_path)
    for file_name in os.listdir(new_files_path[image_class]):
        if not cmp(file_name[len(file_name) - 4:len(file_name)],'.txt'):
            des2 = np.loadtxt(new_files_path[image_class] + file_name)
            match_num = sift_macth_num(des1, des2) 
            match_degree = float(match_num) / float(d_des1)
            if (match_degree - match_rate) > 0: 
                match_file_names.append(file_name[0:len(file_name) - 5] + '.jpg')

    return match_file_names
    

if __name__ == "__main__":
#=======================================================================
#========================创建sift描述子
#=======================================================================    
    print u'创建sift描述子表单...' 
    '''
    img = cv2.imread('./image/Backpack/000-aBFteEPAZNub.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray_img, None)
    
    des = np.array([d/np.linalg.norm(d) for d in des]) #描述子标准化，去除光照影响
    d_des = des.shape[0]
    size_des = len(des.data) / des.shape[0]
    print des, d_des, size_des
    # save_sift_des()
    '''
    d_des1, size_des1, des1 = linux_image_to_des('./image/Backpack/000-aBFteEPAZNub.jpg')
    d_des2, size_des2, des2 = linux_image_to_des('./image/Backpack/000-ACjENmLrpFgO.jpg')
    print sift_macth_num(des1, des1)
    print u'创建结束...'

























