# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:37:35 2017

@author: Young
"""

import numpy as np
import scipy as sp
import pylab as py
from scipy.optimize import minimize

def sigmoid(x):
	#print x,'========='
	#print "======================="
	return 1 / (1 + np.exp(-x))
    
def normalize(X, Ub = 0, Lb = 1, order = 1):
	"""
    归一化
    """
	MAX = 0
	MIN = 0
	X = np.array(X)
	if (order == 0):
		MAX, MIN = X.max(), X.min()
	elif (order == 1):
		MAX, MIN = X.max(1), X.min(1)
	else:
		MAX, MIN = X.max(0), X.min(0)
	
	scale, divd	= (MIN, MAX - MIN)
	if order != 0:
		scale[divd == 0] = 0
		divd[divd == 0] = MAX[divd == 0]
	if (order == 0):
		X = (X - scale) / divd*(Lb - Ub) - Ub
	elif (order == 1):
		X = (X - scale.reshape(-1,1)) / divd.reshape(-1, 1)*(Lb - Ub) - Ub
	else:
		X = (X - scale.reshape(1,-1)) / divd.reshape(1, -1)*(Lb - Ub) - Ub
	return X, scale, divd

def normalize_by_extent(X, scale, divd, Ub = 0,Lb = 1, order = 1):
	if (order == 0):
		X=(X - scale) / divd*(Lb - Ub) - Ub
	elif (order == 1):
		X = (X - scale.reshape(-1, 1)) / divd.reshape(-1, 1)*(Lb - Ub) - Ub
	else:
		X = (X - scale.reshape(1,-1)) / divd.reshape(1,-1)*(Lb-Ub) - Ub
	return X
    
class LogisticRegression:
	"""
    Logistic Regression Class
    Reference: <统计学习方法> 李航
	"""
    
	def __init__(self, X, y, lam = 0.0001, nor = True):
		self.X = np.array(X)
		self.y = np.array(y).flatten(1)	
		(self.N,self.M) = X.shape
		self.nor = nor
		if (nor):
			self.X,self.scale,self.dvi = normalize(self.X)
		assert self.X.shape[1] == self.y.size
		self.lam = lam;
		self.label, self.y = np.unique(y, return_inverse=True)
		self.classNum = self.label.size
		self.theta = np.zeros((self.classNum,self.N)).reshape(self.classNum*self.N)
		self.groundTruth = np.zeros((self.classNum,self.M))
		self.groundTruth[self.y,np.arange(0,self.M)] = 1
		if (y.shape[0] != self.M):
			print "the size of given data is wrong\n"
			return
			
	def LRcost(self, theta):
		#print self.X.shape,theta.reshape(self.classNum,self.N).shape
		theta = theta.reshape(self.classNum,self.N);
		M = np.dot(theta,self.X)
		#print theta.reshape(self.classNum,self.N)
		M = M - M.max()
		h = np.exp(M)
		h = np.true_divide(h, np.sum(h,0))
		#print -np.sum(groundTruth*np.log(h))/self.M
		cost = -np.sum(self.groundTruth*np.log(h))/self.M + self.lam/2.0*np.sum(theta**2);
		grad = -np.dot(self.groundTruth - h,self.X.transpose())/self.M + self.lam*theta;
		grad = grad.reshape(self.classNum*self.N)
		return cost,grad

	def train(self, maxiter=200, disp = False):
		#res,f,d=sp.optimize.fmin_l_bfgs_b(self.LRcost,self.theta,disp=1)
		x0 = np.random.rand(self.classNum,self.N).reshape(self.classNum*self.N)/10
		res = sp.optimize.minimize(self.LRcost,x0, method='L-BFGS-B', jac=True, options = {'disp': disp,'maxiter': maxiter})
		self.theta = res.x
		
	def predict(self, pred):
		if (self.nor):
			pred = normalize_by_extent(pred, self.scale, self.dvi)
		if (pred.shape[0] != self.N):
			print "the data's size for predict is wrong\n"
			print "the process is stop"
			return 
		M = np.dot(self.theta.reshape(self.classNum, self.N), pred)
		h = np.exp(M)
		h = np.true_divide(h, np.sum(h,0)).transpose()
		h = h.argmax(axis = 1)
		return self.label[h]
		
	def output(self):
		print np.dot(self.X, self.theta)