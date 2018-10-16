import numpy as np
from numpy.linalg import inv
from sympy.physics.quantum import TensorProduct
from sympy import *

def swap_col(a, c1, c2):
    a[:,c1],a[:,c2] = a[:,c2],a[:,c1].copy()

def swap_col_r(a, c1, c2):
    t =np.zeros((3,3))
    t[:,c1],t[:,c2] = a[:,c2],a[:,c1].copy()
    return t
def fun(a):
    return 2*a+1

v = [1,2,5,7,9,0,1,0,6]

l = np.count_nonzero(v)
m = np.reshape(v,(3,3))
print (l)
print (m)
print(m.T)
print(inv(m))
print(np.dot(m,inv(m)))
print("=============")
print(m)
swap_col(m,0,2)
print(m)
print(swap_col_r(m,0,2))
print("===Tensor Product == ")
id = Matrix([[1,0],[0,1]])
i = Matrix([[1,0],[-1,0]])
id_i = TensorProduct(id,i)
print(id_i)
print(np.reshape(id_i,(2,8)))
print("===Fun - operation on matrix elemnets")
f_id_i = [ fun(x) for x in id_i]
print(f_id_i)
print("==Reed_Muller Transform==")
w= Matrix([[1,0],[1,1]])
w_1 = w.inv()
R3 = TensorProduct(w,TensorProduct(w,w))
print(R3)
pprint(R3)
f = Matrix([1,0,0,1,1,0,1,1])
pprint(f)
print("===")
s_f = [x%2 for x in R3*f]
pprint(s_f)
print("++++++++++++")
id_5 = np.eye(5)
test = np.reshape([int(x) for x in Matrix(id_5)],(5,5))
pprint(test)

