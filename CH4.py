#coding=utf-8
import numpy as np

a=np.array([1,3,2,4])
b=np.array([-2,1,-1,5])

la=np.sqrt(a.dot(a))
lb=np.sqrt(b.dot(b))
print("----計算向量長度----")
print (la,lb)

cos_angle=a.dot(b)/(la*lb)

print("----計算cos ----")
print (cos_angle)

angle=np.arccos(cos_angle)

print("----計算夾角(單位為π)----")
print (angle)

angle2=angle*360/2/np.pi
print("----轉換單位為角度----")
print (angle2)

# 第三個範例

import numpy as np

a = np.array([[3, 4], [2, 3]])
b = np.array([[1, 2], [3, 4]])
c = np.mat([[3, 4], [2, 3]])
d = np.mat([[1, 2], [3, 4]])
e = np.dot(a, b)
f = np.dot(c, d)
print("----乘法運算----")
print (a * b)
print (c * d)
print("----矩陣相乘----")
print (e)
print (f)

# 第四個範例

import numpy as np

a = np.random.randint(1, 10, (3, 5))

print (a)

# 第五個範例

from numpy import *

a = mat([[1, 2, -1], [3, 0, 1], [4, 2, 1]])

print linalg.det(a)

# 第六個範例

import numpy as np
from matplotlib import pyplot

x = np.arange(0, 10, 0.1)
y = np.sin(x)
pyplot.plot(x, y)
pyplot.show()