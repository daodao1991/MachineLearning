#!/usr/bin/env python3
#-*- coding:utf-8 -*-


from sklearn.naive_bayes import BernoulliNB


def loadDataSet():
    postingList= [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataset):
    '''
    :param dataset: 文档集合
    :return : 包含所有文档中出现的不重复词的列表
    '''
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return : 文档向量
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

