#!/usr/bin/python
# coding=utf-8
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import os
class  estimate:

	"""准确率评估"""
	def Score(self,M,T):
		score = []
		y = T['type']
		x = T.drop(['id','type'],1)
		x_test = x.values
		y_test = y.values
		for i in range(3):
			score.append(M[i].score(x_test,y_test))
		return score

	""" 对三个分类器做评估报告 """
	def class_report(self,M,T):
		y_true = T['type']
		print T
		x = T.drop(['type'],1)
		x_test = x.values
		target = ['class 0','class 1']
		for i in range(3):
			y_pred = M[i].predict(x_test)
			report = classification_report(y_true, y_pred,target_names = target)
			print("分类器{0}的性能报告：\n {1}".format(i,report))

	def roc(self, clf, valid):
		os.chdir(os.path.abspath('..'))
		os.chdir(os.path.abspath('..'))
		path = os.getcwd()

		# X_test = valid.drop(['type','CHROM','POS','PL'],1)
		X_test = valid.drop(['type'], 1)
		y_test = valid['type']
		X_test = X_test.values
		y_score = clf.predict_proba(X_test)
		# Compute ROC curve and ROC area for each class
		fpr, tpr, threshold = roc_curve(y_test, y_score[:, 1])  ###计算真正率和假正率
		roc_auc = auc(fpr, tpr)  ###计算auc的值
		plt.figure()
		lw = 2
		plt.figure(figsize=(10, 10))
		plt.plot(fpr, tpr, color='darkorange',
				 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc="lower right")

		# name =
		plt.savefig(path + '/ROC/ILM_SNP_roc.jpg')
		plt.show()

	def my_confusion_matrix(self,y_true, y_pred):
		from sklearn.metrics import confusion_matrix
		labels = list(set(y_true))
		conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
		print "confusion_matrix(left labels: y_true, up labels: y_pred):"
		print "labels\t",
		for i in range(len(labels)):
			print labels[i], "\t",
		print
		for i in range(len(conf_mat)):
			print i, "\t",
			for j in range(len(conf_mat[i])):
				print conf_mat[i][j], '\t',
			print
		print

	""" 对通过Bagging集成的分类器做评估报告 """
	def bagging_report(self,M,L,LT,U):
		y_true = L['type']
		y_L_true = LT['type']

		y_U_true = U['type']
		target = [' falsely variable','truly  variable']
		y_pred = M.predict(L)
		y_LV_pred = M.predict(LT)
		y_U_pred = M.predict(U)

		self.my_confusion_matrix(y_true, y_pred)  # 输出混淆矩阵
		report0 = classification_report(y_true, y_pred, target_names=target)
		print("三个分类器集成后的性能报告(初始训练集)：\n {0}".format(report0))

		self.my_confusion_matrix(y_L_true, y_LV_pred)  # 输出混淆矩阵
		report1 = classification_report(y_L_true,y_LV_pred,target_names = target)
		print("三个分类器集成后的性能报告(验证集)：\n {0}".format(report1))

		self.my_confusion_matrix(y_U_true, y_U_pred)  # 输出混淆矩阵
		report2 = classification_report(y_U_true, y_U_pred, target_names=target)
		print("三个分类器集成后的性能报告(验证集)：\n {0}".format(report2))






		
