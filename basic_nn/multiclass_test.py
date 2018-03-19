from sklearn import datasets
import numpy as np
from multiclassnn import *
import multiclassnn

iris = datasets.load_iris()
X = np.array(iris.data)
Y = np.array(iris.target)

Y = Y.reshape(150,1)
a = np.zeros((150,3))

a[0:49,0] = 1
a[50:99,1] = 1
a[100:149,2] = 1

print(X.shape,a.shape)

X = X.T

layers_dims = [4, 20, 5, 3] #  4-layer model
parameters = multiclass_nn(X, a.T, layers_dims,learning_rate=0.05, optimizer='gd', num_iterations = 2500, print_cost = True)
