#!usr/bin/env python
#-*- coding:utf-8 -*-

from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape) 

import pylab as pl
pl.gray()
pl.matshow(digits.images[4])
pl.show()
