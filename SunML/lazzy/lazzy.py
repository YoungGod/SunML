# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 15:56:40 2017

@author: Young
"""
import numpy as np

def elu_distance(x_q, X):
    """
    计算查询点x_q与训练样本X中各店的欧拉距离
    """
    return ((x_q - X)**2).sum(axis = 1)

def abs_distance(x_q, X):
    """
    计算查询点x_q与训练样本X中各店的绝对距离
    """
    return (abs(x_q - X)).sum(axis = 1)

def err_evaluation(y_pred,y,err = 'abs'):
    if err == 'square':
        return ((y_pred-y)**2).mean()
    else:
        return (abs(y_pred-y)).mean()
    
class lazzy:
    """
    =======================================
    Lazzy learning
    Reference: 
    A review and comparison of strategies for multi-step ahead time series
    forecasting based on the NN5 forecasting competition
    =======================================
    Input: training set D
           query point x_q, or validation set
           Kmax - the maximum number of neighbors
    Output: y_q - the prediction of the vectorial output of the query point x_q
    
    Steps:
        1. Sort increaingly the set vectors {xi} with respect to the diatance to x_1
        2. [j] will designate the index of the jth closest neighbor of x_q
        3. for k in {2,...,Kmax} do:
            y_qk = sum(y[j])/k
                  # where using [j] collects all the k closest neighbor of x_q
            E_look = e_look.mean()
                  # where e_look = e_k.mean(), e_k = k*(y[j]-y_qk)/(k-1)
            end
        4. K* = arg min{E_look}
        5. y_q = y_qK*
    """
    def __init__(self, Kmax = 50):
        self.Kmax = Kmax
 
    def _lazzy_loo(self, x_q, X, y, dis = elu_distance):
        """
        留一法求给定最大邻居数下，所有备选模型：(误差，邻居数)；
        默认欧拉距离
        """
        l = len(y)
        if dis == elu_distance:
            distance = elu_distance(x_q, X)
        else:
            distance = abs_distance(x_q, X)
        neighbors = distance.argsort()
        models = []    # 用模型记录误差及相应的邻居数
        for k in xrange(2, self.Kmax+1):
            e_look = 0.0
            k_neighbors_idx = neighbors[0:k]
            y_qk = y[k_neighbors_idx].mean(axis = 0)
            for j in k_neighbors_idx:           
                # square err
                e_look += sum((k * (y[j] - y_qk) / (k - 1))**2)
#                # absolute err
#                e_look += sum(abs(k * (y[j] - y_qk) / (k - 1)))
            models.append((e_look/k/l, k))
        return models
    
    def lazzy_prediction(self, x, X, Y, method = 'WIN', dis = 'elu'):
        """
        根据学习的models = lazzy_loo(x, X, Y)，进行预测
        method = 'M'为多模型平均
        method = 'WM'为加权平均
        method = 'WIN'为选择最佳模型
        """
        
        if dis == 'elu':
            distance = elu_distance(x, X)
        else:
            distance = abs_distance(x, X)
        
        models = self._lazzy_loo(x, X, Y, dis = distance)
    
        neighbors_idx = distance.argsort()
        
        if method == 'WIN':
            models.sort()
            num_neighbors = models[0][1]
            y_pred = Y[neighbors_idx[0:num_neighbors]].mean(axis = 0)
            return y_pred
       
        if method == 'M':
            n = len(models)
            y_pred = 0.0
            for err,num_neighbors in models:
                y_pred += Y[neighbors_idx[0:num_neighbors]].mean(axis =0)
            return y_pred/n
        
        if method == 'WM':
            y_pred = 0.0
            total_err = 0.0
            models.sort()
            err_sorted = sorted([err for err, k in models],reverse = True)
            i = 0
            for err,num_neighbors in models:
                y_pred += err_sorted[i] * Y[neighbors_idx[0:num_neighbors]].mean(axis =0)
                total_err += err
                i += 1
            return y_pred/total_err  

if __name__ == '__main__':
    
    def create_dataset(seq, input_lags, pre_period):
        """
        功能：根据时间序列，及给定的输入时滞及预测时长，构建训数据集(X,Y)
        """
        X = []; Y = []
        n = len(seq)
        window = input_lags + pre_period
        for i in xrange(n - window + 1):
    #        # if do like this, you need to pay attention
            x = seq[i: input_lags + i]
    #or     y = seq[input_lags + i: input_lags + pre_period + i]
            y = seq[input_lags + i: window + i]
            
    #        # easy to understand
    #        x_y = seq[i:i+window]
    #        x = x_y[0:input_lags]
    #        y = x_y[input_lags:window]
    
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)
    
    
