# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:33:02 2017

@author: Young
"""

import numpy as np

class RegressionTree:
    """
    re_tree: 回归树
    """
    def __init__(self):

        self.tree = None
        self.pruned_tree = None
   
    def bin_split(self, dataset, feat, value):
        right_set = dataset[np.where(dataset[:,feat] > value)]
        left_set = dataset[np.where(dataset[:,feat] <= value)]
        return right_set, left_set
    
    def err_leaf(self, dataset_y):
        pass
    
    def choose_best_split(self, dataset, tol_m, tol_s):
        """
        功能：中断条件(1) 划分数据集的大小达到限定值
             选择条件：划分后数据集误差减少最大的feature及value
        """
        if dataset[:,-1].std() == 0.0:
            return None, dataset[:,-1].mean()
        
        err = dataset.shape[0]*np.var(dataset[:,-1])
        best_err = np.inf
        num_feat = dataset.shape[1] - 1
        best_feat = 0; best_value = dataset[0,0]
        for i_feat in xrange(num_feat):
            for value in dataset[:,i_feat]:
                right_set, left_set = self.bin_split(dataset, i_feat, value)
                if right_set.shape[0] < 1 or left_set.shape[0] < 1:  # 确保划分数据集不为空
                    continue
                new_err = right_set.shape[0]*np.var(right_set[:,-1]) + left_set.shape[0]*np.var(left_set[:,-1])
                if new_err < best_err:
                    best_feat = i_feat
                    best_value = value
                    best_err = new_err
                    
        right_set, left_set = self.bin_split(dataset, best_feat, best_value)
        if right_set.shape[0] < tol_m and left_set.shape[0] < tol_m:
            return None, dataset[:,-1].mean()
        if err - best_err < tol_s:
            return None, dataset[:,-1].mean()
        
        return best_feat, best_value
    
    def create_tree(self, dataset, tol_m, tol_s):
        """
        功能：创建一个回归树
        """
        best_feat, best_value =  self.choose_best_split(dataset, tol_m, tol_s)
        
        if best_feat == None:    # 通过choose_best_split函数返回值来创建终止条件
            return best_value
        
        tree = {}
        right_set, left_set = self.bin_split(dataset, best_feat, best_value)
        tree['feature'] = best_feat; tree['value'] = best_value                  
        tree['right'] =  self.create_tree(right_set, tol_m, tol_s)
        tree['left'] = self.create_tree(left_set, tol_m, tol_s)
        
        return tree
    
    def train(self, X, y, tol_m = 2, tol_s = 0.0):
#        if len(np.unique(y)) == 1:
        if y.std() == 0.0:
            self.tree = y[0]
        else:
#            print "train1 OK"
            dataset = np.concatenate((X,y.reshape(-1,1)),axis=1)
#            print "trian2 OK"
            self.tree = self.create_tree(dataset, tol_m, tol_s)

    def _pred(self, x, tree):
        """
        """
        feat = tree['feature']
        if x[feat] > tree['value']:
            if type(tree['right']) is dict:
                return self._pred(x, tree['right'])
            else:
                return tree['right']
            
        if x[feat] <= tree['value']:
            if type(tree['left']) is dict:
                return self._pred(x, tree['left'])
            else:
                return tree['left']
    
    def predict(self, X, prune = None):
        if prune == 'pruned':
            tree = self.pruned_tree
            print "pruned"
        else:
            print "nooo"
            tree = self.tree
        try:
            X.shape[1]  # 如果X不是二维array则索引错误，进入except
            y_pred = []
            for x in X:
                y_pred.append(self._pred(x, tree))
            return np.array(y_pred)
        except IndexError:
            return np.array(self._pred(X, tree))
    
    def is_tree(self, tree):
        if type(tree).__name__ == 'dict':
            return True
        else:
            return False
        
    def prune(self, tree, X, y):
        """
        功能：后剪枝
        """
        dataset = np.concatenate((X,y.reshape(-1,1)), axis = 1)
        self.pruned_tree = self._prune(tree, dataset)
    
    def _prune(self, tree, dataset):
        right_set, left_set = self.bin_split(dataset, tree['feature'], tree['value'])
        
        if self.is_tree(tree['right']):
            tree['right'] = self._prune(tree['right'], right_set)
        if self.is_tree(tree['left']):
            tree['left'] = self._prune(tree['left'], left_set)
            
#        print 'right:',tree['right']
#        print 'left:',tree['left']    
        
        if not self.is_tree(tree['right']) and not self.is_tree(tree['left']):
            err_nomerge = sum(np.power(right_set[:,-1]-tree['right'],2)) + \
                          sum(np.power(left_set[:,-1]-tree['left'],2))
            err_merge = sum(np.power(dataset[:,-1]-(tree['right']+tree['left'])/2.0,2))
            if err_merge <= err_nomerge:
                print 'merging'
                return (tree['right']+tree['left'])/2.0
            else:
                return tree
            
        return tree
    
if __name__ == "__main__":
#    np.random.seed(0)
#    X = np.random.random((5,4))
#    y = np.random.randint(0,2,5)
#    re_tree = RegressionTree()
#    re_tree.train(X,y)
#    pred = re_tree.predict(X)
#    print re_tree.tree

    
    # Fitting
    
    fr = open('data.txt','r')
    data_set = []
    for line in fr.readlines():
        data_set.append([float(i) for i in line.strip().split()])
    fr.close()
    
    data_set = np.array(data_set)
    X = data_set[:,0:-1] 
    y = data_set[:,-1]
    
    re_tree = RegressionTree()
    re_tree.train(X,y,tol_m = 4, tol_s = 0.01)
    fit = re_tree.predict(X)
    
    import matplotlib.pyplot as plt
    plt.scatter(data_set[:,1],data_set[:,2],c='b',label = "Origin")
    plt.scatter(data_set[:,1],fit,c='r',label = "Fit")
    plt.legend()
    plt.show()
    
    fit = fit[data_set[:,1].argsort()]
    data_set = data_set[data_set[:,1].argsort(),:]
    plt.plot(data_set[:,1],data_set[:,2],'b-',label = "Origin")
    plt.plot(data_set[:,1],fit,'r-.',label = "Fit")
    plt.legend()
    plt.show()

    # test
    
    fr = open('test_data.txt','r')
    test_data = []
    for line in fr.readlines():
        test_data.append([float(i) for i in line.strip().split()])
    fr.close()
    
    test_data = np.array(test_data)
    X_val = test_data[:,0:-1] 
    y_val = test_data[:,-1]
    pred = re_tree.predict(X_val) 
    
    err_no_prune = sum(np.power(test_data[:,-1] - pred.flatten(),2))
    
    pred = pred[test_data[:,1].argsort()]
    test_data = test_data[test_data[:,1].argsort(),:]
    plt.plot(test_data[:,1],test_data[:,2],'b-',label = "Origin")
    plt.plot(test_data[:,1],pred,'r-.',label = "Predict")
    plt.legend()
    plt.show()
    
    # prune
    import copy
    
    fr = open('test_data.txt','r')
    test_data = []
    for line in fr.readlines():
        test_data.append([float(i) for i in line.strip().split()])
    fr.close()
    
    test_data = np.array(test_data)
    X_val = test_data[:,0:-1] 
    y_val = test_data[:,-1]

    tree = copy.deepcopy(re_tree.tree)
    re_tree.prune(tree, X_val, y_val)
    pred = re_tree.predict(X_val, prune='pruned')
    
    err_pruned = sum(np.power(test_data[:,-1] - pred.flatten(),2))
    
    pred = pred[test_data[:,1].argsort()]
    test_data = test_data[test_data[:,1].argsort(),:]
    plt.plot(test_data[:,1],test_data[:,2],'b-',label = "Origin")
    plt.plot(test_data[:,1],pred,'r-.',label = "Predict-pruned")
    plt.legend()
    plt.show()
    
    
    print err_no_prune
    print err_pruned