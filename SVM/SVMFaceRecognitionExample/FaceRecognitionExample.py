#!/usr/bin/env python
#-*- coding:utf-8 -*-

from time import time
import logging
import matplotlib.pyplot as plt


from sklearn.svm import SVC


print(__doc__)

#将程序进展的信息打印出来
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


######################################################################
#下载数据集，fetch_lfw_people()是一个用来专门用来下载一个
#名人数据集(Labeled Faces in the Wild people dataset)的函数

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)  #lfw_people类似于一个字典结构

n_samples, h, w = lfw_people.images.shape

X = lfw_people.data         #X是一个matrix，每一行是一个实例
n_features = X.shape[1]     #得到每一个实例的维度


y = lfw_people.target       #每一个实例对应的标记,这个例子中就是对应着不同的名人
target_names = lfw_people.target_names
n_classes = target_names.shape[0]  #返回行数，对应了一共有多少类

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###########################################################################
#通过train_test_split()将数据集拆分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


###########################################################################
#PCA降维


n_components = 150

print("Extracting the top %d eigenfaces from %d faces" %
      (n_components, X_train.shape[0]))    #从训练集中的脸部中，提取出前多少张特征脸
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pac.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)  #转化为降维之后的矩阵
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


##########################################################################
#训练一个SVM分类器模型

























