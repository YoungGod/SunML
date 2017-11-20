# coding: UTF-8
"""
Created on Wed May 17 22:46:56 2017

@author: Young
"""
from __future__ import division
import numpy as np
import scipy as sp

class WeakClassfier:
	def __init__(self, X, y):
		"""
        X is the sample
        y is the target
        The weak classifier is a decision Stump
        Reference: <统计学习方法>
		"""
		self.X = np.array(X)
		self.y = np.array(y)
		self.N = self.X.shape[0]
        
	def train(self, W, steps=100):
		"""
		W is a vector with length N. 
		"""
		#print W
		min = 100000000000.0
		t_val = 0;
		t_point = 0;
		t_b = 0;
		self.W = np.array(W)
		for i in range(self.N):
			q ,err = self.findmin(i, 1, steps)
			if (err < min):
				min = err
				t_val = q
				t_point = i
				t_b = 1
		for i in range(self.N):
			q ,err = self.findmin(i, -1, steps)
			if (err < min):
				min = err
				t_val = q
				t_point = i
				t_b = -1
		self.t_val = t_val
		self.t_point = t_point
		self.t_b = t_b
		print self.t_val, self.t_point, self.t_b
		return min
        
	def findmin(self,i,b,steps):
		t = 0
		now = self.predintrain(self.X, i, t, b).transpose()
		err = np.sum((now! = self.y)*self.W)
		#print now
		pgd = 0
		buttom = np.min(self.X[i,:])
		up = np.max(self.X[i,:])
		mins = 1000000;
		minst = 0
		st = (up-buttom)/steps
		for t in np.arange(buttom, up, st):
			now  =  self.predintrain(self.X, i, t, b).transpose()
			#print now.shape,self.W.shape,(now! = self.y).shape,self.y.shape
			err  =  np.sum((now! = self.y)*self.W)
			if err < mins:
				mins = err
				minst = t
		return minst, mins
        
	def predintrain(self, test_set, i, t, b):
		test_set = np.array(test_set).reshape(self.N,-1)
		gt  =  np.ones((np.array(test_set).shape[1],1))
		#print np.array(test_set[i,:]*b)<t*b
		gt[test_set[i,:]*b<t*b] = -1
		return gt

	def pred(self,test_set):
		test_set = np.array(test_set).reshape(self.N,-1)
		t  =  np.ones((np.array(test_set).shape[1],1))
		t[test_set[self.t_point,:]*self.t_b < self.t_val*self.t_b] = -1
		return t
	