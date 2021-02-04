# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:05:55 2021

@author: com
"""
import numpy as np
class Adam:
    
    def __init__(self):
        
        self.mom1 = 0
        self.mom2 = 0
        self.beta1 = 0.99
        self.beta2 = 0.999
        self.count = 1
    def update(self, dW):
        self.mom1 =   self.beta1 * self.mom1   +   (1 - self.beta1) * dW
        self.mom2 =   self.beta2 * self.mom1   +   (1 - self.beta2) * dW * dW
        
        unbias1 = self.mom1 / ( 1 - self.beta1 ** self.count)
        unbias2 = self.mom2 / ( 1 - self.beta2 ** self.count)
        self.count += 1
        
        return unbias1 / ( np.sqrt(unbias2) + 1e-7 )

class SgdM:
    
    def __init__(self , rho = 0.99):
        self.vx = 0.0
        self.rho = rho
    def update(self, dW):
        self.vx = self.rho * self.vx + dW
        
        return self.vx