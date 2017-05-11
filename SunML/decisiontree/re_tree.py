# -*- coding: utf-8 -*-
"""
Created on Mon May 08 14:28:49 2017

@author: Young
"""
import numpy as np

class re_tree:
    """
    re_tree: 回归树
    """
    def __init__(self):

        self.tree = None    
   
    def bin_split(self, dataset, feat, value):
        right_set = dataset[np.where(dataset[:,feat] > value)]
        left_set = dataset[np.where(dataset[:,feat] <= value)]
        return right_set, left_set
    
    def choose_best_split(self, dataset):
        """
        功能：再coding！！！
        """
        err = np.var(dataset[:,-1])
        num_feat = dataset.shape[1] - 1
        for i_feat in xrange(num_feat):
            for value in dataset[:,i_feat]:
                right_set, left_set = self.bin_split(dataset, i_feat, value)
                if np.var(right_set[:,-1]) + np.var(left_set[:,-1]):
                    pass
        return
    
    def create_tree(self, dataset):
        """
        功能：创建一个回归树
        """
        if dataset[:,-1].std == 0:
            return dataset[:,-1][0]
        
        tree = {}
        best_feat, best_value =  self.choose_best_split(dataset) 
        tree['feature'] = best_feat; tree['value'] = best_value
            
        right_set, left_set = self.bin_split(X, best_feat, best_value)
        tree['right'] =  self.create_tree(right_set)
        tree['left'] = self.create_tree(left_set)
        
        return tree
    
    def train(self, X, y):
#        if len(np.unique(y)) == 1:
        if y.std() == 0.0:
            self.tree = y[0]
        else:
            dataset = np.concatenate((X,y.reshape(-1,1)),axis=1)
            self.tree = self.create_tree(dataset)
    
    def predict(self, X):
        pass
    
if __name__ == "__main__":
    X = np.random.random((3,3))
    y = np.array([1,0,1])
    Tree = re_tree()