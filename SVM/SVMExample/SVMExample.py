#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pylab as pl
from sklearn import svm


#创建40个可区分的点
##设置随机数生成器的种子，保证每次运行该文件时产生的数据相同
np.random.seed(0)
##np.r_[]用于连接两个矩阵，要求行数相等
##np.random.randn()是从标准正态分布中，按照指定维度返回一个或多个样本值
X = np.r_[np.random.randn(20, 2)-[2, 2], np.random.randn(20, 2)+[2, 2]]
Y = [0]*20 + [1]*20

#产生SVM模型
clf = svm.SVC(kernel = 'linear')
clf.fit(X, Y)

#得到间隔超平面
##超平面方程：wT*x + b = 0,先得到w
w = clf.coef_[0]
#直线的截距式:y = a*x + b,将超平面的w0*x0 + w1*x1 + w2 = 0转换为截距式
a = -w[0]/w[1]
xx = np.linspace(-5, 5)
#clf.intercept_[0]相当于超平面方程：wT*x + b = 0中的b
yy = a*xx - (clf.intercept_[0])/w[1]


#得到并画出通过支持向量的超平面
##取第一个支持向量
vec1 = clf.support_vectors_[0]
##通过y = a*x + b，求出通过该支持向量的超平面的偏移量b
##进而得到，在该超平面上的一系列点
yy_down = a*xx + (vec1[1] - a*vec1[0])
##取第二个支持向量
vec2 = clf.support_vectors_[-1]
yy_up = a*xx + (vec2[1] - a*vec2[0])


print("support_vectors_:", clf.support_vectors_)
print("clf.coef_:", clf.coef_)

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=100, facecolors='y')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
