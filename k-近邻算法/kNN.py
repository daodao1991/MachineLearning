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

def readfile2Mat(filename):
    myfile = open(filename)
    #读取整个文件得到一个字符串列表，每一行为一个字符串
    listLines = myfile.readlines()
    #删除前n行数据(标题),n自己定
    for i in range(5):
        listLines.remove(listLines[0])
    #得到文件行数
    numbersOfLines = len(listLines)
    #创建一个矩阵，元素初始化为0
    returnMat = zeros((numbersOfLines,3))
    #类标签向量
    labelVector = []   
    count = 0
    for line in listLines:
        #去掉该行字符串两边的空格
        line = line.strip()
        #以制表符\t为分隔符切片string，将整行数据分割成一个元素列表
        listFromLine = line.split('\t\t\t')
        #将列表中的每个字符串型元素转换为浮点数
        listFromLine2 = [float(i) for i in listFromLine]
        #将该行数据的前三个数据赋值给对应的矩阵行
        returnMat[count,:] = listFromLine2[0:3]
        #将该行的最后一个数据，添加到对应的类标签位置
        labelVector.append(int(listFromLine2[-1]))
        count += 1
    return returnMat,labelVector
