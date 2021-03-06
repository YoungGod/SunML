# -*- coding: utf-8 -*-
"""
Created on Wed May 14 19:02:02 2017

@author: Young
"""
from __future__ import division
import numpy as np
import scipy as sp
def heap_judge(a,b):
	return a>b

class Heap:
	def __init__(self,K=None,compare=heap_judge):
		"""
		'K' : 堆的大小
		'compare': 比较大小函数
		"""
		self.K=K
		self.compare=compare
		self.heap=['#']
		self.counter=0
        
	def insert(self,a):
        """
        function: 插入操作
        """
		#print self.heap
		#if self.K!=None:
		#	print a.x,'==='
		if self.K==None:
			self.heap.append(a)
			self.counter+=1
			self.up(self.counter)
		else:
			if self.counter<self.K:
				self.heap.append(a)
				self.counter+=1
				self.up(self.counter)
			else:
				if (not self.compare(a,self.heap[1])):
					self.heap[1]=a
					self.down(1)
		return
        
	def up(self,index):
		if (index==1):
			return
		'''
		print index
		for t in range(index+1):
			if t==0:
				continue
			print self.heap[t].x
		print 
		'''
		if self.compare(self.heap[index],self.heap[int(index/2)]):
			#fit the condition
			self.heap[index],self.heap[int(index/2)]=self.heap[int(index/2)],self.heap[index]
			self.up(int(index/2))
		return
        
	def down(self,index):
		if 2*index>self.counter:
			return
		tar_index=0
		if 2*index<self.counter:
			if self.compare(self.heap[index*2],self.heap[index*2+1]):
				tar_index=index*2
			else:
				tar_index=index*2+1
		else:
			tar_index=index*2
		if not self.compare(self.heap[index],self.heap[tar_index]):
			self.heap[index],self.heap[tar_index]=self.heap[tar_index],self.heap[index]
			self.down(tar_index)
		return

	def delete(self,index):
        """
        function: 删除操作
        """
		self.heap[index],self.heap[self.counter]=self.heap[self.counter],self.heap[index]
		self.heap.pop()
		self.counter-=1
		self.down(index)
		pass

	def delete_ele(self,a):
		try:
			t=self.heap.index(a)
		except ValueError:
			t=None
		if t!=None:
			self.delete(t)
		return t