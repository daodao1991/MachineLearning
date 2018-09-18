#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

#注意此处导入的模块，原来的cross_validation和grid_search都已经弃用了，换成了
#model_selection模块；同时，原来的RandomsizedPCA也弃用了，换成PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


#print(__doc__)

#将程序进展的信息打印出来
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


######################################################################
#下载数据集，fetch_lfw_people()是一个用来专门用来下载一个
#名人数据集(Labeled Faces in the Wild people dataset)的函数

#lfw_people类似于一个字典结构
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4) 

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

#从训练集中的脸部中，提取出前多少张特征脸
print("Extracting the top %d eigenfaces from %d faces" %
      (n_components, X_train.shape[0])) 
t0 = time()
pca = PCA(svd_solver='randomized', n_components=n_components, whiten=True).fit(X_train)
#pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)  #转化为降维之后的矩阵
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


##########################################################################
#训练一个SVM分类器模型

print("Fitting the calssifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid) ##建立分类器
clf = clf.fit(X_train_pca, y_train)  ##进行建模 
print("done ine %0.3fs"% (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

##########################################################################
#在测试集上对模型进行评估
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs"% (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


##########################################################################
#使用matplotlib定性评估预测
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8*n_col, 2.4*n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()       
