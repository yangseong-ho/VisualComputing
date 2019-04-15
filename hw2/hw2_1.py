import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sympy import *

from sympy import solve

# 퍼셉트론 알고리즘을 계산하는 클래스

class Perceptron():

    # 초기설정값들 Critical_Points, Learning_Rate, L_iter 값을 설정한다
    # 임계점, 학습률, 학습횟수
    def __init__(self, Critical_Points=0.0, Learning_Rate=0.01, L_iter=10, w_ = []):

        self.Critical_Points = Critical_Points

        self.Learning_Rate = Learning_Rate

        self.L_iter = L_iter

        if(w_ == []):

            self.w_ = np.zeros(3)

        else:

            self.w_ = w_

 

    # 함수 구현부분

    def fit(self,X,y):

        self.errors_ = []

        self.one = np.ones(len(X))

        self.y_ = list(zip(self.one, X[:,0], X[:,1]))

        self.y_ = np.array(self.y_)

 

        for _ in range(self.L_iter):

            errors = 0

            for xi, target,i in zip(X,y,range(0,len(X))):

                update = (target - self.predict(xi))

                self.w_ += self.Learning_Rate * update * self.y_[i]

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

 

# 모든 데이터들의 산점도를 그린다

plt.plot(data1['x1'],data1['x2'],'ro',data2['x1'],data2['x2'],'bo',data3['x1'],data3['x2'],'go')

plt.grid()

plt.show()

 

 

data12 = np.vstack((data1,data2))

dataC12 = np.hstack((class1,class2))

dataC12 = np.where(dataC12 == 'w1' ,-1,1)

 

data13 = np.vstack((data1,data3))

dataC13 = np.hstack((class1,class3))

dataC13 = np.where(dataC13 == 'w1' ,-1,1)

 

# 초기벡터를 설정한다

initAvec = np.array([0.5,0.5,0.5])

 

# 여러번 설정 값을 바꿔가면서 퍼셉트론 알고리즘을 돌려본다

ptron1 = Perceptron(Learning_Rate = 0.1)

ptron2 = Perceptron(Learning_Rate = 1)

ptron3 = Perceptron(Learning_Rate = 0.5, w_ = initAvec)

 

ptron1.fit(data12, dataC12)

print("\n")

ptron2.fit(data12, dataC12)

print("\n")

ptron3.fit(data12, dataC12)

print("\n")

 

def frange(x, y, jump):

  while x < y:

    yield x

    x += jump

# 퍼셉트론 알고리즘을 통해 나온 x, y 값을 통해 (-10,10)구간에서 그래프를 그려본다

yy = []

for i in frange(-10,10,0.1):

    yy.append(0.8879*i  - 1.1206)

 

xx = np.linspace(-10,10,201)

plt.title('[w1, w2] Learning_Rate=0.1, a=[0.5,0.5,0.5]',fontsize=16)

plt.xlabel('x1')

plt.ylabel('x2')

plt.plot(xx,yy) 

plt.plot(data1['x1'],data1['x2'],'ro',data2['x1'],data2['x2'],'bo')

plt.legend(['perceptron','w1','w2'])

plt.grid()

plt.show()


# 퍼셉트론 알고리즘을 통해 나온 x, y 값을 통해 (-10,10)구간에서 그래프를 그려본다

yyy = []

for i in frange(-10,10,0.1):

    yyy.append(0.8879*i  - 1.1206)

 

xx = np.linspace(-10,10,201)

plt.title('[w1, w2] Learning_Rate=1, a=[0.5,0.5,0.5]',fontsize=16)

plt.xlabel('x1')

plt.ylabel('x2')

plt.plot(xx,yyy) 

plt.plot(data1['x1'],data1['x2'],'ro',data2['x1'],data2['x2'],'bo')

plt.legend(['perceptron','w1','w2'])

plt.grid()

plt.show()

 

 

 

# 퍼셉트론 알고리즘을 통해 나온 x, y 값을 통해 (-10,10)구간에서 그래프를 그려본다

yyy2 = []

for i in frange(-10,10,0.1):

    yyy2.append(0.83319*i - 1.5583)

 

xx = np.linspace(-10,10,201)

plt.title('[w1, w2] Learning_Rate=0.5, a = [0.5,0.5,0.5]',fontsize=16)

plt.xlabel('x1')

plt.ylabel('x2')

plt.plot(xx,yyy2) 

plt.plot(data1['x1'],data1['x2'],'ro',data2['x1'],data2['x2'],'bo')

plt.legend(['perceptron','w1','w2'])

plt.grid()

plt.show()