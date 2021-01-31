# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:33:55 2021

@author: 손익준
"""

import cupy as np

class TwoLayerCNN:
    
    def __init__(self):
        self.W = None
    def train(self , X , y ,learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=100):

        # ReLU류 활성화 함수를 위한 가중치 초기화
        if self.W = None:
            W = np.random.randn(fan_in,fan_out)/np.sqrt(fan_in/2)
        # Xavier initialzation 사용
        N , =X.shape
    def predict():
    