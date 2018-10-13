#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivate(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        '''
        :parameter layers: A list
        包含每一层网络的神经元个数，元素个数代表层数
        :parameter activation: 激励函数，可以为"tanh"或"logistic"
        '''
        if activation == 'logistic':
            self.activation = logistic
            self.activation_derivative = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivate

        self.weights = []
        for i in range(1, len(layers)-1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
    
            
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        #epochs代表训练的次数
        X = np.atleast_2d(X)  
        temp = np.ones([X.shape[0], X.shape[1]+1]) #初始化一个全一矩阵
        temp[:, 0:-1] = X   #添加一个偏项
        X = temp
        y = np.array(y) 

        for k in range(epochs):
            row = np.random.randint(X.shape[0])
            a = [X[row]]
        
            for l in range(len(self.weights)):
                #对每一层完成正向的更新，完成后a中就存放了各个层中单元的输出值
                a.append(self.activation(np.dot(a[l], self.weights[l])))

            error = y[i] - a[-1] #真实值-预测值
            deltas = [error*self.activation_derivative(a[-1])]
            #此时算出了输出层的误差


            for l in range(len(a)-2, 0, -1): #从倒数第二层到第0层，倒着往回走
                deltas.append(deltas[-1].dot(self.weights[l].T))*self.activation_derivative(a[l])
            deltas.reverse() #此时deltas中就存放了隐藏层和输出层的误差
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)


    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
