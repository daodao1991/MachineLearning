#!/usr/bin/env python
#-*- coding:utf-8 -*-

from sklearn.feature_extraction import DictVectorizer
import csv


'''
读取csv文件，并将数据转换为特征列表和标记列表的形式，因为必须将数据转换为
相应的格式，才能使用sklearn库中的决策树分类器DecisionTreeClassifier()
'''
#首先使用DictVectorizer进行特征向量化
allElectronicsData = open(r'E:\Python_jobs\MachineLearning\DecisionTree\AllElectronics.csv','r')
reader = csv.reader(allElectronicsData)  #reader()会按行读取这个文件对象
#header = reader.next()  此为Python2中的用法
#Python3中的用法如下
headers = next(reader)  #此时headers即为第一行内容的一个列表，也就是把特征值的title存入headers这个变量中，以方便
                        #接下来创建特征值的字典使用


print(reader)
print(headers)

#创建两个列表，特征列表和标记列表
featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])  #将每行的最后一个元素即标记信息，添加到labelList中
    rowDict = {}                       #创建一个字典用以存储每行特征值
    for i in range(1, len(row)-1):     #从第2列到倒数第二列，即特征值的范围
        rowDict[headers[i]] = row[i]   #！！！为每一行的字典添加键值对
    featureList.append(rowDict)
    
print(featureList)  #此时featureList列表中的元素为，每一行特征值对应的字典

# 向量化特征列表
vec = DictVectorizer()                 #注意DictVectorizer()的用法
dummyX = vec.fit_transform(featureList).toarray()

print("dummyX: " + '\n' + str(dummyX))
print(vec.get_feature_names())

