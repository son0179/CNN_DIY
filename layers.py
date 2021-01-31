# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:41:52 2021

@author: 손익준
"""

import numpy as np

class ConvLayer:
    def __init__(self, X, W, B , params):
        
        self.w = np.ones((7,3,4,4))
        self.B = B
        self.params = params 
        
    def conv_forward(self, x , w , B , params):
        # 초기값 설정
        N, C, H, W = x.shape
        F, C, FH , FW = w.shape
        stride = params["stride"]
        pad = params["pad"]
        if ( self.W == None):
        outH = (H + 2 * pad - FH) // stride + 1
        outW = (W + 2 * pad - FW) // stride + 1
        
        # 이미지를 행렬 데이터로 변환 im2col
        
        img = np.pad(x, ( (0,0),(0,0) ,(pad,pad),(pad,pad) ) , 'constant'  )
        col = np.zeros( (N, C, FH, FW, outH, outW ) )
        
        for i in range(FH):
            endi = stride * outH + i
            for j in range(FW):
                endj = stride * outW  + j           
                col[:,:,i,j,:,:] = img[:,:, i:endi:stride , j:endj:stride ]
        
        col = np.transpose(col ,(0,4,5,1,2,3))
        col = col.reshape(N*outH*outW,-1)
        
        
        print(col.shape)
        
        w = w.reshape((F,-1)).T
        
        
        print(w.shape)
        
        out = np.dot( col , w)
        
        print(out.shape)
        
        out = out.reshape(N, outH, outW , -1 ).transpose(0,3,1,2)
        
        print(out.shape)
        return out
    
 """   
    def cov_backward(self, dout):
        
        N, C, H, W = x.shape
        F, C, FH , FW = w.shape
        
        dout  = dout.transpose(0,2,3,1).reshape(-1,)

"""    
class AffineLayer:
    
    def __init__(self):
        self.w = np.ones((7,3,4,4))
    def aff_forward(X,b):
        
        

def ReLu(X):
    
    out = X[X<0]
    cach = X
    return out, X