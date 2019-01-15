import matplotlib.pyplot as plt
import numpy as np

a = np.array([[1,2,3,4],
     [5,6,7,8]])

print(a)

print(a[1:])

x= [0,1,2,3,4,5,6,7,8,9]
y= [1,2,1.2,3,2,2,1,9,0.2,3.2]
plt.plot(x,y)
plt.show()