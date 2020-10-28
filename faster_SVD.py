import gzip
import json
import ast
import numpy as np
import pandas as pd

row_sum = 200
col_sum = 500
arr  = np.zeros((10000, 5000))
#generate a special case, with given row_sum and col_sum
for i in range(row_sum):
    arr.ravel()[i::arr.shape[1]+row_sum] = 1
np.random.shuffle(arr)
A = arr# A is the reqd matrix
# for future uses we keep the following calculated already
k = 100
l = k+20
m = 10000
n = 5000

# now in the first step we choose k = 100, l = 102, i = 2, A is considered mxn , so m = 10000, n= 5000
# In the first step, Using a random number generator, form a real n × l matrix G whose entries are independent and identically distributed
# Gaussian random variables of zero mean and unit variance
mu = 0
sigma = 1.0
G = np.random.normal(mu, sigma, (n,l))
#Compute B = AG (B ∈ R^ m×l )
B = np.matmul(A,G)#---------------------------------------------------------one multiplication by A

X, lamda, Yt = np.linalg.svd(B, full_matrices=True)

Q = X[:, 1 : k] #(Q ∈ R m×k )
Qt = Q.transpose()

T = np.matmul(Qt,A)#-------------------------------------------------------2nd mult by A, only these two are there in this algo
W, singlr, Vt = np.linalg.svd(T, full_matrices=True)

U = np.matmul(Q,W)
final = np.multiply(U,singlr)
final
df = pd.DataFrame (final)
filepath = 'output.xlsx'
df.to_excel(filepath, index=False)

#https://www.mathworks.com/help/matlab/matlab_prog/resolving-out-of-memory-errors.html
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

# Now we implement the same algo as above but the input matrix is created ans stored in a sparse matrix format and hence speeds up dot and
# multiplication. 

import random
#data = np.ones(1000)
# making sure that the elems in each row and cols are unique will guarnatee that elems in matrix are bin
# %reset
def createRandomSortedList(num, start, end): 
    arr = [] 
    tmp = random.randint(start, end) 
      
    for x in range(num): 
          
        while tmp in arr: 
            tmp = random.randint(start, end) 
              
        arr.append(tmp) 
    return arr 
myrow = createRandomSortedList(10, 1, 1000)
mycol = createRandomSortedList(10, 1, 1000)
myrow,mycol



import gzip
import json
import ast
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from numpy.random import rand
from scipy.sparse import csr_matrix, isspmatrix_csr


row  = np.array([199, 157, 675, 429, 892, 532, 669, 650, 204, 931])
col  = np.array([362, 793, 543, 669, 639, 659, 92, 859, 177, 404])
newdata = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
data = np.array(newdata)
A = coo_matrix((data, (row, col)), shape=(1000, 1000)).tocsr()
At = coo_matrix((data, (col, row)), shape=(1000, 1000)).tocsr()
k = 100
l = k+20
m = 1000
n = 1000
# now in the first step we choose k = 100, l = 102, i = 2, A is considered mxn , so m = 10000, n= 5000
# In the first step, Using a random number generator, form a real n × l matrix G whose entries are independent and identically distributed
# Gaussian random variables of zero mean and unit variance
mu = 0
sigma = 1.0
G = np.random.normal(mu, sigma, (n,l))
#Compute B = AG (B ∈ R^ m×l )
#B = np.matmul(A,G)#---------------------------------------------------------one multiplication by A
B = []
for i in range(l):
    v = G[:,i]
    x = A.dot(v)
    B.append(x)
B = np.array(B).T

X, lamda, Yt = np.linalg.svd(B, full_matrices=True)

Q = X[:, : k] #(Q ∈ R m×k )
Qt = Q.transpose()

#T = np.matmul(Qt,A)#-------------------------------2nd mult by A, only these two are there in this algo
T = []
for i in range(k):
    v = G[:,i]
    x = At.dot(v)
    T.append(x)
T = np.array(T)
W, singlr, Vt = np.linalg.svd(T, full_matrices=True)

U = np.matmul(Q,W)
final = np.multiply(U,singlr)

final
#df = pd.DataFrame (final)
#filepath = 'output.xlsx'
#df.to_excel(filepath, index=False)
