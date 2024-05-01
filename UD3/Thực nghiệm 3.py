import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import time
import math
# chon cac gia tri n,r trong thuat toan
n = 2
r = 1
# Ma tran A
A = np.array([[0,0],[0,-8]])
# chon rho > 0 trong thuat toan
p = 1
# vecto b
b = np.array([1,0])
# ma tran don vi trong R^n
I = np.identity(n)
res = np.array([[-1/8,math.sqrt(63)/8],[-1/8,-math.sqrt(63)/8], [-1,0]])
# cong thuc tim hinh chieu
def hc(x):
  if LA.norm(x) > r:
    x = r*x/LA.norm(x)
  return x
# Công thức sai số er
def er(x,k):
    if LA.norm(x[k%2]) > 1:
        return LA.norm(x[(k+1)%2] - x[k%2])/LA.norm(x[k%2])
    else:
        return LA.norm(x[(k+1)%2] - x[k%2])
    
# Thuat toan DC
k = 0
X = [ np.random.randint(-1000,1001,size = (n) ), np.random.randint(-1000,1000, size = (n)) ]
while er(X,k) >= 1e-6:
    X[(k+1)%2] = hc(  1/p*np.dot(X[k%2], p*I - A) - b/p)
    k = k + 1
# In ra kết quả hội tụ và khoảng cách tới nghiệm gần nhất
S = '(' + str(X[k%2][0]) + ',' + str(X[k%2][1]) +  ')'
print("Điểm hội tụ là:", S)
print("Khoảng cách tới nghiệm gần nhất là:", min(LA.norm(X[k%2] -res[0]),LA.norm(X[k%2] -res[1]),LA.norm(X[k%2] -res[2])))
