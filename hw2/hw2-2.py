import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sympy import *

from sympy import solve

# 데이터를 불러온 다음 클래스별로 구분한다

projectData = '2D_Sample_Pattern.txt'

names = ['x1', 'x2', 'class']

data = pd.read_csv(projectData, header=None, names=names)

 

Class = data['class']

Data = np.array(data)

 

data1 = data[0:10]

data2 = data[10:20]

data3 = data[20:30]

 

class1 = data1['class']

class2 = data2['class']

class3 = data3['class']

 

del data['class']

del data1['class']

del data2['class']

del data3['class']

data12 = np.vstack((data1,data2))

dataC12 = np.hstack((class1,class2))

dataC12 = np.where(dataC12 == 'w1' ,-1,1)

 

data13 = np.vstack((data1,data3))

dataC13 = np.hstack((class1,class3))

dataC13 = np.where(dataC13 == 'w1' ,-1,1)

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

# BatchRelaxationWithMargin 알고리즘을 계산하는 클래스 

class BatchRelaxationWithMargin():

    # 초기설정값들 Critical_Points, Learning_Rate, L_iter, margin 값을 설정한다
    # 임계점, 학습률, 학습횟수, 마진

    def __init__(self, Critical_Points=0.0, Learning_Rate=0.01, L_iter=10, margin = 0.1):

        self.Critical_Points = Critical_Points

        self.Learning_Rate = Learning_Rate

        self.L_iter = L_iter

        self.margin = margin

 

    # 함수 구현부분

    def fit(self,X,y):

        self.w_ = np.zeros(1 + X.shape[1])

        self.errors_ = []

        self.one = np.ones(len(X))

        self.y_ = list(zip(self.one, X[:,0], X[:,1]))

        self.y_ = np.array(self.y_)

 

        for _ in range(self.L_iter):

            errors = 0

            for xi, target,i in zip(X,y,range(0,len(X))):

                update = (target - self.predict(xi))

                if(update != 0):

                    delta = (update * self.y_[i] * (self.margin - self.w_ * update * self.y_[i] )) / ((np.linalg.norm(update * self.y_[i]))**2)

                    self.w_ += delta * self.Learning_Rate

                errors += int(update != 0.0)

            self.errors_.append(errors)

            print(self.w_)

        return self

 

    # x*w + b를 구현한 함수

    def net_input(self, X):

        return np.dot(X, self.w_[1:]) + self.w_[0]

 

    # 임계값을 초과하는 예측 클래스를 반환하는 함수

    def predict(self, X):

        return np.where(self.net_input(X) > self.Critical_Points, 1, -1)

bat = BatchRelaxationWithMargin(Learning_Rate = 0.1, margin = 0.1)

bat.fit(data13, dataC13)

print("\n")

bat2 = BatchRelaxationWithMargin(Learning_Rate = 0.5, margin = 0.5)

bat2.fit(data13, dataC13)

print("\n")

bat3 = BatchRelaxationWithMargin(Learning_Rate = 0.2, margin = 0.1)

bat3.fit(data13, dataC13)

# 알고리즘을 통해 나온 x, y 값을 통해 (-10,10)구간에서 그래프를 그려본다

yy2 = []

xx = np.linspace(-10,10,201)

for i in frange(-10,10,0.1):

    yy2.append(0.1668*i + 1.119)

 

plt.title('[w1, w3] Learning_Rate=0.1, margin = 0.1',fontsize=16)

plt.xlabel('x1')

plt.ylabel('x2')

plt.plot(xx,yy2) 

plt.plot(data1['x1'],data1['x2'],'ro',data3['x1'],data3['x2'],'go')

plt.legend(['Batch Relaxation with margin','w1','w3'])

plt.grid()

plt.show()

 

 

# 알고리즘을 통해 나온 x, y 값을 통해 (-10,10)구간에서 그래프를 그려본다

yy3 = []

xx = np.linspace(-10,10,201)

for i in frange(-10,10,0.1):

    yy3.append(0.6091*i + 0.8582)

 

plt.title('[w1, w3] Learning_Rate=0.5, margin = 0.5',fontsize=16)

plt.xlabel('x1')

plt.ylabel('x2')

plt.plot(xx,yy3) 

plt.plot(data1['x1'],data1['x2'],'ro',data3['x1'],data3['x2'],'go')

plt.legend(['Batch Relaxation with margin','w1','w3'])

plt.grid()

plt.show()

 

# 알고리즘을 통해 나온 x, y 값을 통해 (-10,10)구간에서 그래프를 그려본다

yy7 = []

xx = np.linspace(-10,10,201)

for i in frange(-10,10,0.1):

    yy7.append(0.3437*i + 0.6732)

 

plt.title('[w1, w3] Learning_Rate=0.2, margin = 0.1',fontsize=16)

plt.xlabel('x1')

plt.ylabel('x2')

plt.plot(xx,yy7) 

plt.plot(data1['x1'],data1['x2'],'ro',data3['x1'],data3['x2'],'go')

plt.legend(['Batch Relaxation with margin','w1','w3'])

plt.grid()

plt.show()