# -*- coding: utf-8 -*-
"""
Created on Mon May 08 14:28:49 2017

@author: Young
"""
import numpy as np

class RegressionTree:
    """
    re_tree: 回归树
    """
    def __init__(self):

        self.tree = None    
   
    def bin_split(self, dataset, feat, value):
        right_set = dataset[np.where(dataset[:,feat] > value)]
        left_set = dataset[np.where(dataset[:,feat] <= value)]
        return right_set, left_set
    
    def err_leaf(self, dataset_y):
        pass
    
    def choose_best_split(self, dataset):
        """
        功能：中断条件(1) 划分数据集的大小达到限定值
             选择条件：划分后数据集误差减少最大的feature及value
        """
        err = dataset.shape[0]*np.var(dataset[:,-1])
        num_feat = dataset.shape[1] - 1
        best_feat = 0; best_value = dataset[0,0]
        for i_feat in xrange(num_feat):
            for value in dataset[:,i_feat]:
                right_set, left_set = self.bin_split(dataset, i_feat, value)
                if right_set.shape[0] < 1 or left_set.shape[0] < 1:  # 隐含最小划分粒度为2
                    continue
                new_err = right_set.shape[0]*np.var(right_set[:,-1]) + left_set.shape[0]*np.var(left_set[:,-1])
                if new_err < err:
                    best_feat = i_feat
                    best_value = value
                    err = new_err
        return best_feat, best_value
    
    def create_tree(self, dataset):
        """
        功能：创建一个回归树
        """
        tol_m = 2
        tol_err = 0.00
        err = dataset.shape[0]*np.var(dataset[:,-1])
        
        if dataset[:,-1].std() == 0:
            return dataset[:,-1][0]
        
        if dataset.shape[0] < tol_m:
            return dataset[:,-1].mean()
        
        best_feat, best_value =  self.choose_best_split(dataset)
        right_set, left_set = self.bin_split(dataset, best_feat, best_value)
        
        new_err = right_set.shape[0]*np.var(right_set[:,-1]) + left_set.shape[0]*np.var(left_set[:,-1])
        err_reduced = err - new_err
        print "create tree OK1", err_reduced
        
        
        if err_reduced < tol_err:
#            print "create tree leaf", err_reduced
            return dataset[:,-1].mean()
        
        tree = {}
#        right_set, left_set = self.bin_split(X, best_feat, best_value)
        tree['feature'] = best_feat; tree['value'] = best_value                  
        tree['right'] =  self.create_tree(right_set)
        tree['left'] = self.create_tree(left_set)
        
        return tree
    
    def train(self, X, y):
#        if len(np.unique(y)) == 1:
        if y.std() == 0.0:
            self.tree = y[0]
        else:
#            print "train1 OK"
            dataset = np.concatenate((X,y.reshape(-1,1)),axis=1)
#            print "trian2 OK"
            self.tree = self.create_tree(dataset)
    
    def predict(self, X):
        pass
    
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.random((5,4))
    y = np.random.randint(0,2,5)
    re_tree = RegressionTree()
    re_tree.train(X,y)
    print re_tree.tree