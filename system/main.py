#!/usr/bin/python
# coding=utf-8
import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
from Tri_training01 import Tri_training01
from Tri_training02 import Tri_training02
from Tri_training03 import Tri_training03
from Tri_training04 import Tri_training04
import Bagging
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import confusion_matrix


"""准确率评估"""
def Score(M,T):
    score = []
    y = T['type']
    x = T.drop(['id','type'],1)
    x_test = x.values
    y_test = y.values
    for i in range(3):
        score.append(M[i].score(x_test,y_test))
    return score

""" 对三个分类器做评估报告 """
def class_report(M,T):
    y_true = T['type']

    x = T.drop(['type'],1)
    x_test = x.values
    target = ['class 0','class 1']
    for i in range(3):
        y_pred = M[i].predict(x_test)
        report = classification_report(y_true, y_pred,target_names = target)
        print("分类器{0}的性能报告：\n {1}".format(i,report))
def roc_computer(clf,reader):
    X_test = reader.drop(['type'], 1)
    y_test = reader['type']
    X_test = X_test.values
    y_score = clf.predict_proba(X_test)
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_test, y_score[:, 1])  ###计算真正率和假正率
    return fpr, tpr, threshold

def roc(tri04_clt,tri02_clt,tri03_clt,tri01_clt, reader04,reader03,reader02,reader01):

    # Compute ROC curve and ROC area for each class
    fpr04, tpr04, threshold04 = roc_computer(tri04_clt, reader04)
    roc_auc04 = auc(fpr04, tpr04)  ###计算auc的值
    fpr03, tpr03, threshold03 = roc_computer(tri03_clt, reader03)
    roc_auc03 = auc(fpr03, tpr03)  ###计算auc的值
    fpr02, tpr02, threshold02 = roc_computer(tri02_clt,reader02)
    roc_auc02 = auc(fpr02, tpr02)  ###计算auc的值
    fpr01, tpr01, threshold01 = roc_computer(tri01_clt, reader01)
    roc_auc01 = auc(fpr01, tpr01)  ###计算auc的值


    lw = 2
    # plt.figure(figsize=(10, 10))
    # 设置输出的图片大小
    figsize = 10, 10
    figure, ax = plt.subplots(figsize=figsize)
    # plt.figure(figsize=(10, 10))
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23,
             }

    # 设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }

    plt.plot(fpr04, tpr04, color='fuchsia',
             lw=lw, label='NA12878(area = %0.2f)' % roc_auc04)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot(fpr03, tpr03, color='blue',
             lw=lw, label='ILM_INDEL_Test_stander(area = %0.2f)' % roc_auc03)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot(fpr02, tpr02, color='darkorange',
             lw=lw, label='NA12878-GATK3-chr21_2000(area = %0.2f)' % roc_auc02)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot(fpr01, tpr01, color='lime',
             lw=lw, label='NA12877(area = %0.2f)' % roc_auc01)  ###假正率为横坐标，真正率为纵坐标做曲线


    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',font2)
    plt.ylabel('True Positive Rate',font2)
    plt.title('ROC Curve',font2)
    plt.legend(loc="lower right")

    plt.savefig('C:/Users/Sherwin/Desktop/newpicture/roc.eps',dpi=1200)

def my_confusion_matrix(y_true, y_pred):

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

def class_reporter(clf,reader,name):
    target = [' falsely variable', 'truly   variable']
    X_test = reader
    y_test = reader['type']
    y_test_pre = clf.predict(X_test)
    my_confusion_matrix(y_test, y_test_pre)  # 输出混淆矩阵
    report = classification_report(y_test, y_test_pre, target_names=target)
    print(name+"   性能报告(测试集)：\n {0}".format(report))

def precision_recall_computer(clf,reader):
    X_test = reader.drop(['type'], 1)
    y_test = reader['type']
    X_test = X_test.values
    # Compute precision and recall for each class
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])
    return precision, recall, thresholds

def X_Y(S):
    train = S
    y = train['type']
    x = train.drop(['type'], 1)
    return x, y

def precision_recall(tri04_clt,tri02_clt,tri03_clt,tri01_clt, reader04,reader03,reader02,reader01):
    precision_04, recall_04, thresholds_04 = precision_recall_computer(tri04_clt, reader04)
    precision_03, recall_03, thresholds_03 = precision_recall_computer(tri03_clt, reader03)
    precision_02, recall_02, thresholds_02 = precision_recall_computer(tri02_clt, reader02)
    precision_01, recall_01, thresholds_01 = precision_recall_computer(tri01_clt, reader01)

    x_test04, y_test04 = X_Y(reader04)
    x_test03, y_test03 = X_Y(reader03)
    x_test02, y_test02 = X_Y(reader02)
    x_test01, y_test01 = X_Y(reader01)

    ap_04 = average_precision_score(y_test04, tri04_clt.predict_proba(x_test04)[:, 1])
    ap_03 = average_precision_score(y_test03, tri03_clt.predict_proba(x_test03)[:, 1])
    ap_02 = average_precision_score(y_test02, tri02_clt.predict_proba(x_test02)[:, 1])
    ap_01 = average_precision_score(y_test01, tri01_clt.predict_proba(x_test01)[:, 1])

    figsize = 10, 10
    figure, ax = plt.subplots(figsize=figsize)
    # plt.figure(figsize=(10, 10))
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    plt.plot(precision_04, recall_04, color='fuchsia', lw=2,
             label='NA12878(average = %0.2f)' % ap_04)
    plt.plot(precision_03, recall_03, color='lime', lw=2,
             label='ILM_INDEL_Test_stander(average = %0.2f)' % ap_03)
    plt.plot(precision_02, recall_02,color='darkorange',lw=2,
             label='NA12878-GATK3-chr21_2000(average = %0.2f)' % ap_02)
    plt.plot(precision_01, recall_01, color='blue', lw=2,
             label='NA12877(average = %0.2f)' % ap_01)
    # print "Average precision of ap_02: {:.3f}".format(ap_02)
    # print "Average precision of ap_03: {:.3f}".format(ap_03)
    # print "Average precision of ap_03: {:.3f}".format(ap_01)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Precision',font2)
    plt.ylabel('Sensitivity',font2)
    plt.title('Precision-Sensitivity Curve', font2)
    plt.legend(loc=4)
    plt.savefig("C:/Users/Sherwin/Desktop/newpicture/Precision-Sensitivity Curve.eps",dpi=1200)

""" 对通过Bagging集成的分类器做评估报告 """
def bagging_report(tri04_clt,tri03_clt,tri02_clt,tri01_clt,reader04,reader03,reader02,reader01):
    name04 = "NA12878"
    class_reporter(tri04_clt, reader04, name04)
    name03 = "ILM_INDEL_Test_stander"
    class_reporter(tri03_clt,reader03, name03)
    name02 = "NA12878-GATK3-chr21_2000"
    class_reporter(tri02_clt, reader02, name02)
    name01 = "NA12877"
    class_reporter(tri01_clt,reader01, name01)

    roc(tri04_clt,tri02_clt,tri03_clt,tri01_clt, reader04,reader03,reader02,reader01)
    precision_recall(tri04_clt,tri02_clt,tri03_clt,tri01_clt, reader04,reader03,reader02,reader01)


if __name__ == "__main__":
    data04 = pd.read_csv('G:/NewProject02_new_best/Alldata/labeled/NA12878_stander.csv')
    data03 = pd.read_csv('G:/NewProject02_new_best/Alldata/labeled/ILM_INDEL_Test_stander.csv')
    data02 = pd.read_csv('G:/NewProject02_new_best/Alldata/labeled/NA12878-GATK3-chr21_2000.csv')
    data01 = pd.read_csv('G:/NewProject02_new_best/Alldata/labeled/NA12877_stander.csv')

    tri04 = Tri_training04.Tri_training04()
    tri04.pre_data()
    M04 = tri04.Training()
    tri04_clt = Bagging.Bagging(M04)
    print "tri04 结束，tri03 开始"
    print "/n"
    tri03 = Tri_training03.Tri_training03()
    tri03.pre_data()
    M03 = tri03.Training()
    tri03_clt = Bagging.Bagging(M03)
    print "tri03 结束，tri02 开始"
    print "/n"
    tri02 = Tri_training02.Tri_training02()
    tri02.pre_data()
    M02 = tri02.Training()
    tri02_clt = Bagging.Bagging(M02)
    print "tri02 结束，tri01 开始"
    print "/n"
    tri01 = Tri_training01.Tri_training01()
    tri01.pre_data()
    M01 = tri01.Training()
    tri01_clt = Bagging.Bagging(M01)

    reader04 = data04.drop(['CHROM', 'POS', 'PL'], 1)
    reader02 = data02.drop(['CHROM', 'POS', 'PL'], 1)
    reader03 = data03
    reader01 = data01.drop(['CHROM', 'POS', 'PL'], 1)
    print "性能报告"
    bagging_report(tri04_clt,tri03_clt,tri02_clt,tri01_clt,reader04,reader03,reader02,reader01)

