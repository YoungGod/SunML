# -*- coding: utf-8 -*-
"""
Created on Mon May 08 14:28:49 2017

@author: Young
"""
import numpy as np

class CART:
    """
    CART:分类和回归树
    """
    def __init__(self, X, y, indicator = None):
        
        self.X = np.array(X)
        self.y = np.array(y)
        
        if indicator == None:
            self.indicator = np.zeros(len(y))
        else:
            self.indicator = indicator
            
    def train(self):
        pass
    
    def predict(self, X):
        pass
    
if __name__ == "__main__":
    X = np.random.random((3,3))
    y = np.array([1,0,1])
    Tree = CART(X, y)