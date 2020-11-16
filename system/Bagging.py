#!/usr/bin/python
# coding=utf-8
import numpy as np


""" 以硬投票的方式集成三个已训练好的分类器 """

class  Bagging:
	""" 构造函数参数为一个含有三个分类器的列表 """
	def __init__(self, arg):
		self.M = arg  # 初始的三个分类器

	""" 预测函数 """
	def predict(self,T):
		x = T.drop(['type'],1)
		x_test = x.values
		y_1 = self.M[0].predict(x_test)
		y_2 = self.M[1].predict(x_test)
		y_3 = self.M[2].predict(x_test)
		y = np.array([y_1,y_2,y_3])
		y_pre = list(map(sum,zip(*y)))
		def f(x):
			if x>1:
				return 1
			else:
				return 0
		y_pre = list(map(f,y_pre))

		y_1_p = self.M[0].predict_proba(x_test)
		y_2_p = self.M[1].predict_proba(x_test)*3
		y_3_p = self.M[2].predict_proba(x_test)
		y_p = y_1_p + y_2_p+y_3_p
		# print a
		y_pr = y_p.argmax(axis=1)
		return y_pr
		# return np.array(y_pre)

	def predict_proba(self,unlabelData):

		x_test = unlabelData

		y_1_p = self.M[0].predict_proba(x_test)
		y_2_p = self.M[1].predict_proba(x_test)*3
		y_3_p = self.M[2].predict_proba(x_test)
		y_p = (y_1_p + y_2_p+y_3_p)/3
		return y_p