## Created by : Nilay

# Python file of all the activation functions used

import numpy as np


def sigmoid(Z):
    return (1/(1+np.exp(-(Z)))), Z

def relu(Z):
    return np.maximum(0,Z), Z

def softmax(Z):

    ##   This is the normal softmax
    # denominator = np.sum(np.exp(Z), axis = 0)
    # return Z/denominator, Z

    ##  For numerical stability so that the values don't overshoot, I'll use this stable softmax
    ##  softmax(X) = softmax(X+C); C = -max(Xi)
    C = - np.max(Z, axis = 0)
    exps = np.exp(Z + C)
    denominator = np.sum(exps, axis = 0)
    return exps/denominator, Z
