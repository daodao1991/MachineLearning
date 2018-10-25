#!usr/bin/env python
#-*- coding:utf-8 -*-

#手写数字的识别
# 每个图片8x8  识别数字：0,1,2,3,4,5,6,7,8,9

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split

digits = load_digits() #把数据集装载进来
X = digits.data
y = digits.target
X -= X.min() # 将所有的值转化到0-1之间
X /= X.max()

nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print("start fitting")
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
    
