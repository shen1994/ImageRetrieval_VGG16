# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 09:48:07 2017

@author: shenxinfeng
"""


import numpy as np

def euclidean_distance(vec1, vec2):
    vec_dot = np.dot((vec1-vec2), (vec1-vec2).T)
    return np.sqrt(vec_dot)

def cosine(vec1, vec2):
    vec_dot = np.dot(vec1, vec2.T)
    vec_A = np.sqrt(np.dot(vec1, vec1.T))
    vec_B = np.sqrt(np.dot(vec2, vec2.T))
    return float(vec_dot)/float(vec_A * vec_B)

def chebyshev_distance(vec1, vec2):
    return np.max(np.abs(vec1 - vec2))
    
def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1-vec2))
