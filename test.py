from numpy.random._common import namedtuple
from sklearn import preprocessing
import pandas as pd
import numpy
from numpy import *

import random
positive_training_bags = [1,2,3,4,5,6,7,8,9,10,11,12,13]
c=numpy.zeros(len(positive_training_bags))
print(c)
m=list(map(lambda x: x[0]+x[1],zip(positive_training_bags,c)))
print(m)
print(list(map(lambda x: x/2,c)))
k = 10
random_positive_bags = list(map(lambda x: positive_training_bags[x], range(0, k)))
total_instances = sum([4 for bag in random_positive_bags])

c = numpy.full(10, 0.1)
print(numpy.negative(positive_training_bags))

def min_max_scaler(tp_reader):
    X_tt = tp_reader.iloc[:,1:10]

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_tt)

    df = pd.DataFrame(X_train_minmax,columns = ['nhet','cnlr.median','mafR','mafR.clust','start','end','cf.em',	'tcn.em','lcn.em'])
    # df['id'] = tp_reader['id']
    df.insert(0,'id',tp_reader['id'])
    df['label'] = tp_reader['label']
    # feature = df.reindex(columns=['id','nhet','cnlr.median','mafR','mafR.clust','start','end','cf.em',	'tcn.em','lcn.em','label'], fill_value=1)
    # X_test_minmax = min_max_scaler.transform(X_test)
    temp_path = 'F:/shenzhen/Sherwin/HRD/all_scaler.csv'
    df.to_csv(temp_path,index=False, sep=',')


# tpDataFile = 'F:/shenzhen/Sherwin/HRD/all.csv'
# tp_reader = pd.read_csv(tpDataFile, low_memory=False)
# min_max_scaler(tp_reader)
def bubble_sort(arr):
    length = len(arr)
    while length > 0:
        for i in range(length - 1):
            if arr[i][1] > arr[i + 1][1]:
                arr[i] = arr[i] + arr[i + 1]
                arr[i + 1] = arr[i] - arr[i + 1]
                arr[i] = arr[i] - arr[i + 1]
        length -= 1
    return arr
def quick_sort(arr, low, high):
    i = low
    j = high
    if i >= j:
        return arr
    temp = arr[i]
    while i < j:
        while i < j and arr[j][0].accuracy >= temp[0].accuracy:
            j = j - 1
        arr[i] = arr[j]
        while i < j and arr[i][0].accuracy <= temp[0].accuracy:
            i = i + 1
        arr[j] = arr[i]
    arr[i] = temp
    quick_sort(arr, low, i - 1)
    quick_sort(arr, j + 1, high)
    return arr


PredictionResult = namedtuple('PredictionResult', [
    'accuracy',
    'precision',
    'recall'
])


if __name__ == "__main__":
    c1 = PredictionResult(
        accuracy=6,
        precision=2,
        recall=3
    )
    c2 = PredictionResult(
        accuracy=2,
        precision=3,
        recall=4
    )
    c3 = PredictionResult(
        accuracy=7,
        precision=3,
        recall=4
    )
    a = [[c1,2],[c2,3],[c3,4]]
    print(a)
    c = quick_sort(a, 0, len(a) - 1)
    print(c)
    m =[[1,3],[2,6],[3,6]]
    n_h = []
    for i in m:
        n_h.append(i[1])
    print(n_h)

    total = 0

    list1 = [11, 5, 17, 18, 23]

    for ele in range(0, len(list1)):
        total = total + list1[ele]

    print("列表元素之和为: ", mean(list1))