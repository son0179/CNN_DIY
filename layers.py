# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:41:52 2021

@author: 손익준
"""

import numpy as np

class ConvLayer:
    def __init__(self, Wshape , params , B = 0.1 , learning_rate = 1e-7):
        #self.w = np.ones(Wshape) * 1e-3
        self.w = np.random.randn(Wshape[0],Wshape[1],Wshape[2],Wshape[3])/np.sqrt(Wshape[2]/2)
        self.b = B
        self.params = params
        self.lr = learning_rate
        
    def conv_forward(self, x ):
        # 초기값 설정
        w=self.w
        self.xshape = x.shape
        N, C, H, W = x.shape
        F, C, FH , FW = w.shape
        stride = self.params["stride"]
        pad = self.params["pad"]
        #if ( self.w == None):
        #    pass
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
        
        w = w.reshape((F,-1)).T
        self.col_x = col
        self.col_w = w
        
        out = col @ w
        
        out = out.reshape(N, outH, outW , -1 ).transpose(0,3,1,2)
        
        
        return out
    

    def conv_backward(self, dX ):
        
        # = self.
        w = self.w
        N, C, H, W =self.xshape
        F, C, FH , FW = w.shape
        stride = self.params["stride"]
        pad = self.params["pad"]
        
        dX = dX.transpose(0,2,3,1)
        dX = dX.reshape(-1,F)
        
        dW = dX.T @ self.col_x
        dW = dW.reshape(F,C,FH,FW)
        
        dcol = dX @ self.col_w.T
        
        outH = (H + 2 * pad - FH) // stride + 1
        outW = (W + 2 * pad - FW) // stride + 1
        # 행렬을 이미지로 변환 col2im
        col = dcol.reshape(N, outH, outW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)
    
        img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
        for i in range(FH):
            endi = stride * outH + i
            for j in range(FW):
                endj =stride * outW + j
                img[:, :, i:endi:stride, j:endj:stride] += col[:, :, i, j, :, :]
        
        dx = img[:, :, pad:H + pad, pad:W + pad]
        
        return dx , dW


class AffineLayer:
    
    def __init__(self , learning_rate = 1e-7):
        self.ck = None
        self.w = None
        self.lr = learning_rate
        
        
    def aff_forward(self, x):
        self.xshape = x.shape
        self.x = x.reshape(x.shape[0],-1)
        
        if self.ck == None:
            #self.w = np.ones((self.x.shape[1],10)) * 1e-3
            self.w = np.random.randn(self.x.shape[1],10)/np.sqrt(self.x.shape[1]/2)
            self.ck = 1
            
        out = self.x @ self.w #+ b
        return out
    
    
    def aff_backward(self,dW):  
        
        W = np.array(self.w)
        X = np.array(self.x)
        
        
        dX = dW @ W.T
        dX = dX.reshape(self.xshape)
        
        #dB = dW.T @ np.ones(X.shape[0])
        
        dW = X.T @ dW
        return dX, dW #, dB



class ReLULayer:
    
    def __init__(self):
        self.X = None
        
    def ReLU_forward(self , x ):
        self.X = x
        out = np.maximum(0, x)
        return out
    
    def ReLU_backward(self,x):  
        out = np.array(x)
        out[self.X<=0] = 0
        
        return out




def softmax(x,y):
    x = np.exp(x - np.max(x , axis=1 ,keepdims = True)) #softmax 내에서 exp 했을때 값의 overflow을 방지
    x /= np.sum(x,axis=1 , keepdims = True)             # 결과 값은 동일
    dW = np.array(x)
    
    ans_score = x[np.arange(x.shape[0]),y]
    
    loss= np.sum(ans_score) / (x.shape[0] )
    loss = np.log(loss)
    loss *= -1
    
    dW[np.arange(x.shape[0]), y] -= 1
    dW /= x.shape[0]

    

    return loss , dW