#!/usr/bin/python
# coding=utf-8
import warnings

warnings.filterwarnings("ignore")
import warnings

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
from sklearn.externals import joblib

from system.Tri_training01.process import Process

from system.Tri_training01.Learn import Learn
from SherwinDemo.Test.RWCSV_LOH import RWCSV
import pandas as pd
import math
from system.Tri_training01.bootstrapSample import Sample

from system.Tri_training01.Report import estimate
from system.Tri_training01.Bagging import Bagging
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.model_selection import train_test_split


class Tri_training01:
    """三体训练法"""

    def __init__(self):
        self.L = None  # 原始已标记样本集
        self.Un = None  # 初始未标记样本集
        self.M = []  # 初始三个分类器
        self.M0 = []  # 初始三个分类器
        self.T = None  # 测试集
        self.Sn = None  # 初始三个训练集
        self.Learn = None
        self.e = [0.7, 0.7, 0.7]
        self.l_1 = [0, 0, 0]
        self.Ln = []
        self.LT = []
        self.LV = []
        self.reader = []
        self.Update = []
        self.E = []
        self.UE = []
        self.UU = []
        self.UN = []
        self.data = []
        self.train = []
        # self.S_T=None
        self.Estimate = estimate()  # 评估器
        self.Result = open('C:/Users/Sherwin/Desktop/result0.2.txt', 'ab')

    """ 获取X,Y """

    def X_Y(self, S):
        train = S
        y = train['type']
        x = train.drop(['type'], 1)
        x_train = x.values
        y_train = y.values.ravel()
        return x, y

    """ 准备数据 """

    def pre_data(self):
        RW = RWCSV()
        # ds_file_mut2_30X = 'C:/Users/Sherwin/Desktop/newdata/DATA/fp.csv'
        # dp_file_mut2_30X = 'C:/Users/Sherwin/Desktop/newdata/DATA/dp.csv'
        # RW.meger(ds_file_mut2_30X, dp_file_mut2_30X)
        data = 'C:/Users/Sherwin/Desktop/LOH_HRD/lable_result/train.tsv'
        u_data = 'C:/Users/Sherwin/Desktop/LOH_HRD/lable_result/unlabel.tsv'

        train_set, unabled_set = RW.pre_data(data)
        uu_reader = pd.read_csv(u_data,sep='\t')
        self.UU = uu_reader
        # self.UU = uu_reader.drop(['id'], 1)
        self.T = unabled_set
        pro = Process()
        pro.read_labeled(train_set, unabled_set,uu_reader)
        # pro.read_unlabeled()
        self.L = pro.L
        self.UN = pro.U

        # self.reader = pro.L
        # unlabel = pro.T
        unlabel = pro.U
        self.UE = unlabel

        self.Un = unlabel.drop(['type'], 1)


        # self.U = pro.T.drop(['CHROM', 'POS', 'SAMPLEID', 'REF', 'ALT', 'AF0', 'DP'], 1)
        # X, y = self.X_Y(self.reader)
        # X_test, y_test = self.X_Y(self.U)

        # self.Ln = pro.TV
        xt, yt = self.X_Y(self.L)
        # X_test1, y_test1 = self.X_Y(self.Ln)
        # X_train, X_tv, y_train, y_tv = train_test_split(X, y, test_size=0.70, random_state=2)
        X_train, X_validation, y_train, y_validation = train_test_split(xt, yt, test_size=0.1,random_state=2,stratify=yt)
        # learn = Learn()
        # self.Learn = learn
        # for i in range(3):
        # 	clf = learn.reModel(i)
        # 	clf.fit(X_train, y_train)
        # 	print ('Score', clf.score(X_train, y_train), clf.score(X_tv, y_tv))
        # 	self.M.append(clf)
        # tri_clt = Bagging(self.M)
        #
        X_train['type'] = y_train
        self.Sn = X_train
        # X_tv['type'] = y_tv
        # self.T = X_tv
        # # y_pred = tri_clt.predict(self.T)
        # self.Estimate.bagging_report(M=tri_clt, T=self.T, L=self.L)

        train_data = self.Sn
        sample = Sample()
        # sample.gen_Train_Set(train_data)
        # self.train = sample.S

        learn = Learn()
        self.Learn = learn
        for i in range(3):
            clf = learn.reModel(i)
            # xtt, ytt = self.X_Y(self.train[i])
            X_tt,y_tt = self.X_Y(self.Sn)
            clf.fit(X_tt, y_tt)

            print ('Score', clf.score(X_tt, y_tt), clf.score(X_validation, y_validation))

            scores = cross_val_score(clf, X_tt, y_tt, cv=5)  # cv为迭代次数。
            print(scores)  # 打印输出每次迭代的度量值（准确度）
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
            self.M.append(clf)
            # tri_clt0 = Bagging(self.M0)
            #
        X_validation['type'] = y_validation
        self.LT = X_validation
        X_train['type'] = y_train
        self.LV = X_train
        # print "single_mut1_type"
        # self.Estimate.bagging_report(M=tri_clt0, T=self.LT, L=self.Ln)
        # print "over"
        # y_pred = tri_clt0.predict(self.Ln)
        # lnto = pro.TV
        # lnto['PreType'] = y_pred
        # lnto.to_csv('C:/Users/Sherwin/Desktop/tmp/Ln.csv', index=False)
        # joblib.dump(tri_clt0, 'C:/Users/Sherwin/Desktop/Model/tri_clt02.model')

        """ 估计从两个分类器的组合中导出的假设的分类错误率 """

    def MeasureError(self, H_j, H_k):
        count = 0
        err = 0
        labeled = self.LT.values
        L_x = labeled[:, :-1]
        L_y = labeled[:, -1]
        pre_j = H_j.predict(L_x)
        pre_k = H_k.predict(L_x)
        j_eq_k = pre_j == pre_k
        j_eq_k = list(j_eq_k)
        count = len([x for x in j_eq_k if x == True])
        for x in range(len(labeled)):
            if j_eq_k[x] and pre_j[x] != L_y[x]:
                err += 1
        print (err, count)
        err = float(err) / float(count)
        toWrite = "count:" + str(count) + ',' + str(len(labeled)) + '\r\n'
        self.Result.write(toWrite)
        print ("count:", count, len(labeled))
        return err, count

    """ 随机移除指定数目的样本 """

    def Subsample(self, L_t, s):
        y_train = L_t['type']
        X_train = L_t.drop(['type'], 1)
        n = L_t.shape[0] - s
        X_train, X_tv, y_train, y_tv = train_test_split(X_train, y_train, test_size=n, random_state=1)
        print (n, Counter(y_train), 'today')
        X_train['type'] = y_train
        sample = X_train
        L_t.drop(sample.index.values, inplace=True)

    """ 训练过程 """

    def Training(self):
        tri_clt = Bagging(self.M)

        self.Estimate.bagging_report(tri_clt, self.LV, self.LT, self.UE)
        change = True
        n = 0
        while change:
            change = False
            count = [0, 0, 0]
            n += 1
            Train = [None, None, None]
            for i in range(3):
                self.Ln.append(pd.DataFrame())
                self.Update.append(False)
                M_jk = [self.M[x] for x in range(3) if x != i]
                self.E.append(self.MeasureError(M_jk[0], M_jk[1]))

                print ('Sherwin', self.E[i], self.e[i])
                if self.E[i][0] < self.e[i]:
                    iters = self.Un.iterrows()
                    for row in iters:
                        row_x = row[1].values
                        pre_j = M_jk[0].predict(row_x.reshape(1, -1))
                        pre_k = M_jk[1].predict(row_x.reshape(1, -1))
                        if pre_j == pre_k:
                            x = row[1].to_frame().transpose()
                            x['type'] = pre_j
                            self.Ln[i] = self.Ln[i].append(x, ignore_index=True)
                            count[i] += 1
                    if self.l_1[i] == 0:
                        item = (self.E[i][0] / (self.e[i] - self.E[i][0])) + 10
                        self.l_1[i] = math.floor(item)
                    if self.l_1[i] < count[i]:
                        if self.E[i][0] * count[i] < self.e[i] * self.l_1[i]:
                            self.Update[i] = True
                        elif self.l_1[i] > self.E[i][0] / (self.e[i] - self.E[i][0]):
                            s = (self.e[i] * self.l_1[i]) / self.E[i][0] - 1
                            s = int(math.ceil(s))
                            print (self.e[i], self.l_1[i], self.E[i][0])
                            self.Subsample(self.Ln[i], s)
                            count[i] = s
                            self.Update[i] = True
            for i in range(3):
                if self.Update[i] == True:
                    change = True
                    # Train[i] = pd.concat([self.train[i], self.Ln[i]], ignore_index=True)
                    Train[i] = pd.concat([self.Sn, self.Ln[i]], ignore_index=True)
                    X_train, y_train = self.X_Y(Train[i])
                    clf = self.Learn.reModel(i)
                    clf.fit(X_train, y_train)
                    self.M[i] = clf
                    self.e[i] = self.E[i][0]
                    self.l_1[i] = count[i]

            self.Update = []
            self.Ln = []
            self.E = []

            print('第{0}次循环'.format(n))
            tri_clt = Bagging(self.M)  # 分类结果出口

            self.Estimate.bagging_report(tri_clt, self.LV, self.LT, self.UE)
            # 保存模型

        # 预测结果
        tri_clt = Bagging(self.M)
        X_t = self.LT.drop(['type'], 1)
        y_t = self.LT['type']

        data = pd.read_csv('C:/Users/Sherwin/Desktop/LOH_HRD/lable_result02/test/TRUE/T190003020BCD_190003021TD_lableResult_loh.tsv',sep='\t')
        data = data.drop(['chrom','seg','num.mark','segclust','cnlr.median.clust'], 1)
        self.data = data.fillna(data.mean())
        y_pred = tri_clt.predict(self.data)
        y_predpro = tri_clt.predict_proba(self.data)

        self.data['PreType'] = y_pred
        self.data['y_predpro'] = y_predpro
        self.data.to_csv('C:/Users/Sherwin/Desktop/LOH_HRD/lable_result02/test_/TRUE/T190003020BCD_190003021TD_lableResult_loh.tsv', index=False,sep='\t')
        return self.M


if __name__ == "__main__":
    tri = Tri_training01()
    tri.pre_data()
    tri.Training()
