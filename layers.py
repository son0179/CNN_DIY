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
    
    def __init__(self , X ):
        
        self.x = X.reshape(X.shape[0],-1)
        #self.w = np.ones((self.x.shape[1],10)) * 1e-3
        self.w = np.random.randn(self.x.shape[1],10)/np.sqrt(self.x.shape[1]/2)
        
        
    def aff_forward(self, x, b):
        print(self.w.shape, x.shape)
        x = x.reshape((x.shape[0],-1))
        out = x @ self.w + b
        return out
    
    
    def aff_backward(self,dW):  
        W = self.w
        X = self.x
        
        dW = dW @ X
    
        tmp = np.zeros((W.shape[1],X.shape[0]))
        #tmp [y, np.arange(tmp.shape[1])] = 1 
        
        dW -= tmp @ X
        
        dW /= X.shape[0]

class ReLULayer:
    
    def __init__(self):
        self.X = None
        
    def ReLU_forward(self , x ):
        self.X = x
        x[x<0] = 0
        out = x
        return out
    
    def ReLU_backward(self,x):  
        out = np.array(x)
        out = out[self.X<0]
        
        return out


def softmax(x,y):
    x = np.exp(x - np.max(x , axis=1 ,keepdims = True)) #softmax 내에서 exp 했을때 값의 overflow을 방지
    #x /= np.sum(x,axis=1 , keepdims = True)             # 결과 값은 동일
    dW = np.array(x)
    
    ans_score = x[np.arange(x.shape[0]),y]
    
    loss= np.sum(ans_score) / x.shape[0]
    loss = np.log(loss)
    loss *= -1
    
    dW[np.arange(x.shape[0]), y] -= 1
    dW /= x.shape[0]

    

    return loss , dW