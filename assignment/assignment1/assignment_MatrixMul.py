# -*- coding: utf-8 -*-
import numpy as np

class MyMatrix(object):
  def __init__(self):
    pass

  def matmul(self, a, b):
      """ a is the List: m * n
          b is the List  n * s  
          result is the List  m*s """
      m,n,s= len(a),len(a[0]),len(b[0]) 
      if  n!= len(b):
          raise ValueError("shapes ({},{}) and ({},{}) not aligned: {} \
                           (dim 1) != {} (dim 0)".format(m,n,len(b),s,n,len(b)))      
      
      result =[[0 for i in range(m)]for j in range(s)] 
      for i in range(m):       
          for j in range(n):  
              for k in range(s):
                     result[i][k] += a[i][j]*b[j][k] 
      return result

# m*n 
a = [[0,0,0,5],
     [0,1,1,6]]

# n*s
b= [[1,2],
    [3,4],
    [5,6],
    [2,5]]

result_np=np.matmul(a,b) 
print (result_np) 

myMatrix=MyMatrix()
result = myMatrix.matmul(a,b)
print (result)
