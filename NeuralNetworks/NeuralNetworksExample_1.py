#!usr/bin/env python
#-*- coding:utf-8 -*-

from NeuralNetwork import NeuralNetwork
import numpy as np

a = NeuralNetwork([2, 2, 1])
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
a.fit(X,y)
for i in [[0,0], [0,1], [1,0], [1,1]]:
    print(i, a.predict(i))
