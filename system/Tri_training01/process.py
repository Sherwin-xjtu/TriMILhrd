#!/usr/bin/python
# coding=utf-8
import numpy as np
import pandas as pd
from system.Setting.config import Config
from sklearn.preprocessing import MinMaxScaler, Imputer
# from fancyimpute import KNN

# 预处理数据

class Process:
    def __init__(self):
        self.labeled_path = Config.labeled_path
        self.labeled_path = Config.t_labeled_path
        self.unlabeled_path = Config.unlabeled_path
        self.L = None  # 初始已标记样本集
        self.T = None  # 测试集
        self.U = None  # 初始未标记样本集
        self.U_1 = None  # Sherwin
        self.TV = None  # Sherwin 模型验证集

    # 读取已标记数据并且分割训练集和测试集

    def read_labeled(self,train_set,unabled_set,uu_reader):
        label_data = train_set.drop(['chrom','seg','num.mark','segclust','cnlr.median.clust'], 1)
        label_data = label_data.fillna(label_data.mean())
        test_data = unabled_set.drop(['chrom','seg','num.mark','segclust','cnlr.median.clust'], 1)
        test_data = test_data.fillna(test_data.mean())
        # tv_label_data = pd.read_csv('C:/Users/Sherwin/Desktop/LOH_HRD/189003592BCD_189003592TD_fct_cncf.tsv', sep='\t')
        tv_label_data = uu_reader
        tv_label_data = tv_label_data.fillna(tv_label_data.mean())
        scaler = MinMaxScaler()

        # label_data_sample = label_data.sample(n=100,replace=False)
        # train = label_data.drop(['CHROM', 'POS', 'SAMPLEID', 'REF', 'ALT', 'CONTQ', 'TLOD', 'AD'], 1)
        #['CHROM', 'POS', 'ID', 'DP',  'REF', 'ALT', 'SB_rf','SB_rr','SB_af','SB_ar','F1R2_rf','F1R2_af','F2R1_rr','F2R1_ar','MPOS','SBR','FRR','ECNT','CONTQ','TLOD','AF','MMQ','MBQ','STRANDQ','GERMQ','SEQQ','ROQ']
        # train = label_data.drop(['id'], 1)
        train = label_data
        # test = test_data.drop(['id'], 1)
        test = test_data
        tv_label_data = tv_label_data.drop(['chrom','seg','num.mark','segclust','cnlr.median.clust'], 1)
        # tv_label_data = Imputer().fit_transform(tv_label_data)

        scaler.fit(train.iloc[:, :-1])
        train_scaled = scaler.transform(train.iloc[:, :-1])
        test_scaled = scaler.transform(test.iloc[:, :-1])

        tv_label_scaled = scaler.transform(tv_label_data.iloc[:, :-1])
        #
        train_scaled_ = pd.DataFrame(train_scaled, columns=['nhet', 'cnlr.median','mafR','mafR.clust','start','end','cf.em','tcn.em','lcn.em'])
        train_scaled_['type'] = label_data['type'].values

        test_scaled_ = pd.DataFrame(test_scaled, columns=['nhet', 'cnlr.median','mafR','mafR.clust','start','end','cf.em','tcn.em','lcn.em'])
        test_scaled_['type'] = test_data['type'].values

        tv_label_scaled = pd.DataFrame(tv_label_scaled,columns=['nhet', 'cnlr.median', 'mafR', 'mafR.clust', 'start', 'end', 'cf.em','tcn.em', 'lcn.em'])
        tv_label_scaled['type'] = tv_label_data['type'].values
        # print train_scaled_,test_scaled_
        self.L = train
        self.T = test
        self.U = tv_label_data

    # 读取未标记数据
    def read_unlabeled(self):
        unlabel_data = pd.read_csv('G:/NewProject02_new_best/Alldata/labeled/NA12877_stander.csv')
        self.U_1 = unlabel_data
        self.U = unlabel_data


if __name__ == '__main__':
    pro = Process()
    pro.read_labeled()
    pro.read_unlabeled()
    L = pro.L
# print(L)
