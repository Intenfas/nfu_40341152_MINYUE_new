# coding=utf-8
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from scipy.stats import norm
#例子一
fig = plt.gcf()
fig.set_size_inches(10,6)

var('x')
f = lambda x: exp(-x**2/2)

x = np.linspace(-5,6,200)
y = np.array([f(v) for v in x],dtype='float')

plt.grid(True)
plt.title('Gaussian Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x,y,color='gray')
plt.fill_between(x,y,0,color='#c0f0c0')
plt.show()

#例子二
fig, ax = plt.subplots(1, 1)

x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 100)

ax.plot(x, norm.pdf(x),'r-', lw=50, alpha=0.6, label='norm pdf')

ax.plot(x, norm.pdf(x), 'k-', lw=20, label='frozen pdf')

r = norm.rvs(size=1000)

ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)

ax.legend(loc='best', frameon=False)

plt.show()