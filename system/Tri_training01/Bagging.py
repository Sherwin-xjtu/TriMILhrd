#!/usr/bin/python
# coding=utf-8
import numpy as np
from system.Tri_training01.Learn import Learn
from sklearn.metrics import accuracy_score
""" 以硬投票的方式集成三个已训练好的分类器 """

class  Bagging:
	""" 构造函数参数为一个含有三个分类器的列表 """
	def __init__(self, arg):
		self.M = arg  # 初始的三个分类器



	def fit(self,x,y):
		learn = Learn()
		M0 = []
		for i in range(3):
			clf = learn.reModel(i)
			clf.fit(x, y)
			M0.append(clf)
		return M0
	def score(self,x,y):

		return accuracy_score(y, self.predict(x))



	""" 预测函数 """
	def predict(self,T):
		if 'type' in T:
			x = T.drop(['type'],1)
		else:
			x = T
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
		y_2_p = self.M[1].predict_proba(x_test)
		y_3_p = self.M[2].predict_proba(x_test)
		y_p = (y_1_p + y_2_p+y_3_p)/3
		# print a
		y_pr = y_p.argmax(axis=1)
		return y_pr
		# return np.array(y_pre)

	def predict_proba(self,unlabelData):
		if 'type' in unlabelData:
			x = unlabelData.drop(['type'], 1)
		else:
			x = unlabelData

		x_test = x.values
		y_1_p = self.M[0].predict_proba(x_test)
		y_2_p = self.M[1].predict_proba(x_test)
		y_3_p = self.M[2].predict_proba(x_test)
		y_p = (y_1_p + y_2_p+y_3_p)/3
		y_pr = y_p.argmax(axis=1)  #idx
		pro = np.amax(y_p,axis=1)    #每行最大值
		return pro