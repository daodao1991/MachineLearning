#!usr/bin/env python
#-*- coding:utf-8 -*-

from numpy import *
import operator  #运算符模块


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #对距离矩阵按照升序排序，得到排序后的编号
    sortedDistancesIndices = distances.argsort()

    #对前k个最近邻数据所属分类的次数进行统计，统计结果存储在字典claddCount中
    classCount = {}
    for i in range(k):
        aa = labels[sortedDistancesIndices[i]]
        classCount[aa] = classCount.get(aa,0) + 1

    #对字典中的数据按照次数进行降序排序,
    ###先通过items()方法将字典转化为包含键、值对元组的列表
    ###排序时使用了运算符模块中itemgetter函数,用于选择键值对元组中的第二个元素
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    #排序后出现频率最高的那个分类排在第一位
    finalClass = sortedClassCount[0][0]
    
    return finalClass
