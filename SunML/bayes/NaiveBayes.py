# -*- coding: utf-8 -*-
"""
Created on Sat May 06 10:39:08 2017

@author: Young
"""

import numpy as np

class NaiveBayes:
    """
    
    """
    def __init__(self, X, y, indicator = None):
        """
        功能：初始化参数
        输入参数：X,y,indicator(0表示连续变量)
        输出参数：prob_y,prob_xy,mu,sigma
        """
        self.X = np.array(X)  # 训练数据集合应该不是类的属性！！这样设计不合理，修改！！！
        self.y = np.array(y)
        self.prob_y = {}  # 各类别的先验概率
        self.prob_xy = {}  # 各类别下各特征变量相应取值的条件概率
        self.mu = {}
        self.sigma = {}
        
        self._M, self._N = self.X.shape  # _M 样本数量，_N特征数量
        self._C = len(np.unique(self.y))  # 类别数量
        self._labels = np.unique(self.y)  # 类别标签
        
        if indicator is None:
            self.indicator = np.zeros(self._N)
        else:
            self.indicator = indicator
#            print indicator
            
    def train(self):
        """
        功能：学习参数prob_y,prob_xy（离散特征），mu,sigma（连续特征）
        """
        # 计算prob_y
        label_counts = {}
        for j_label in y:
            label_counts.setdefault(j_label,0)
            label_counts[j_label] += 1
        for label in label_counts:
            self.prob_y[label] = float(label_counts[label] + 1) / (self._M + self._C)  # laplce平滑
#        print "Porb_y:", self.prob_y
        
        # 计算prob_xy(离散特征indicator=1)，mu,sigma(连续特征indicator=0)
        counts = {}
        for i_feat in xrange(self._N):
#            print "Feature:", i_feat
            # 离散变量
            if self.indicator[i_feat] == 1:
#                print "Indicator:", indicator[i_feat]
                counts.setdefault(i_feat,{})
                self.prob_xy.setdefault(i_feat,{})
                for value in np.unique(self.X[:,i_feat]):  # 这里通过python特有的值遍历方式，否则采用索引遍历
                    counts[i_feat].setdefault(value,{})
                    self.prob_xy[i_feat].setdefault(value,{})
                    for label in label_counts:
                        counts[i_feat][value].setdefault(label)
                        self.prob_xy[i_feat][value].setdefault(label,0.0)
                        x = self.X[label==y, i_feat]
                        counts[i_feat][value][label] = self._counts(value,x)
                        self.prob_xy[i_feat][value][label] = \
                            float(counts[i_feat][value][label] + 1) / (len(x) + len(np.unique(self.X[:,i_feat])))  # 采用了laplace平滑
            # 连续变量
            else:
                self.mu.setdefault(i_feat,{})
                self.sigma.setdefault(i_feat,{})
                for label in label_counts:
                    self.mu[i_feat].setdefault(label,0.0)
                    self.sigma[i_feat].setdefault(label,0.0)
                    self.mu[i_feat][label] = self.X[:,i_feat].mean()
                    self.sigma[i_feat][label] = self.X[:,i_feat].std()
                
    def _counts(self, value, x):
        """
        功能：统计label下，第i特征，各取值为value的计数
        似乎没有什么意义？ 
        可以把计算prob_xy设计为一个函数
        """
        return np.sum(x == value)
    
    def _pred(self, x):
        """
        功能：给定特征矢量x,及特征变量离散或连续标识indicator，预测所属分类
        """
        best_prob = 0.0
        best_label = self._labels[0]
        x = x.flatten()
        
        prob_yx = {} # 记录各类别下后验概率，用于归一化概率，估计可能性
                     # 后验概率也是条件概率，描述的是给定特征x后，类别y出现的概率
        
#        for i_label in self._labels[i_label]: # 用索引访问对于估计可能性更方便！
        for label in self._labels:  
            prob_c = self.prob_y[label]
            for i_feat in xrange(self._N):
                if self.indicator[i_feat] == 1:
                    for value in x[np.where(indicator == 1)]:  # 采用索引遍历也可以
#                        print x
                        prob_c = prob_c * self.prob_xy[i_feat][value][label]
#                        print prob_c
                else:
                    for value in x[np.where(indicator == 0)]:
                        prob_c = prob_c * self._gauss(value, self.mu[i_feat][label],
                                                      self.sigma[i_feat][label])
#                        print prob_c
#            print label, prob_c

            if prob_c > best_prob:
                best_prob = prob_c
                best_label = label
#                print best_prob, best_label
            prob_yx[label] = prob_c
                
        best_prob = prob_yx[best_label] / sum(prob_yx.values())  # 概率估计
        return best_prob, best_label
    
    def _pred_log(self, x):
        """
        功能：对数概率计算，该方法可以集成在_pred函数中，但为了预测计算更快重写一个函数
        """
        best_prob = -np.inf
        best_label = self._labels[0]
        x = x.flatten()
        
        # log prob
        for label in self._labels:  
            prob_indicator1 = 0
            prob_indicator0 = 0
            for i_feat in xrange(self._N):
                if self.indicator[i_feat] == 1:
                    for value in x[np.where(indicator == 1)]:  
                        prob_indicator1 += np.log(self.prob_xy[i_feat][value][label])
                else:
                    for value in x[np.where(indicator == 0)]:
                        prob_indicator0 += np.log(self._gauss(value, self.mu[i_feat][label],
                                                      self.sigma[i_feat][label]))
                        
            prob_c = np.log(self.prob_y[label]) + prob_indicator1 + prob_indicator0

            if prob_c > best_prob:
                best_prob = prob_c
                best_label = label
        return best_label        
        
    
    def pred(self, X, prob_type = None):
        """
        功能：给定新样本特征向量集X,预测分类
        """
        if prob_type == None:
            prob = []
            label = []
            for x in X:
                best_prob, best_label = self._pred(x)
                prob.append(best_prob)
                label.append(best_label)
            return np.array(label),np.array(prob)
        elif prob_type == 'log':
            label = []
            for x in X:
                best_label = self._pred_log(x)
                label.append(best_label)
            return np.array(label)
    
    def _gauss(self, value, mu, sigma):
        """
        功能：在高斯分布假设下，计算连续特征变量取值value的概率
        """
        e = np.exp(-np.power(value-mu,2) / (2*sigma**2))
        return 1 / np.sqrt(2*np.pi) / sigma * e
    

# test
if __name__ == "__main__":

    np.random.seed(0)
    X = np.random.random((4,3))
    x = np.array([1,0,1,0]).reshape(-1,1)
    X = np.concatenate((X,x),axis = 1)  # axis = 0 为行方向|，axis = 1 为列方向——
    x = np.array([0,1,1,1]).reshape(-1,1)
    
    X = np.concatenate((x,X),axis = 1)
    """
    X = [
    [ 0.        ,  0.5488135 ,  0.71518937,  0.60276338,  1.        ],
    [ 1.        ,  0.54488318,  0.4236548 ,  0.64589411,  0.        ],
    [ 1.        ,  0.43758721,  0.891773  ,  0.96366276,  1.        ],
    [ 1.        ,  0.38344152,  0.79172504,  0.52889492,  0.        ]]
    """
    indicator = np.array([1,0,0,0,1])
    y = np.array(["yes","no","yes","no"])
    
    X = [[1,1],
         [1,0],
         [0,0],
         [0,0]]
    X = np.array(X)
    indicator = np.array([1,1])
    y = np.array(["yes","yes","no","no"])
    
    NB = NaiveBayes(X, y, indicator)
    
    NB.train()
    label, prob = NB.pred(X)
    
    print label
    print prob
    
    label = NB.pred(X, prob_type = 'log')
    print label
##    NB.pred(X[3,:])
#    print "sample 0 is:",NB.pred(X[0,:]), ",real is:",y[0],"\n", \
#        "sample 1 is:",NB.pred(X[1,:]),",real is:",y[1],"\n", \
#        "sample 2 is:",NB.pred(X[2,:]),",real is:",y[2],"\n", \
#        "sample 3 is:",NB.pred(X[3,:]),",real is:",y[3],"\n"










    