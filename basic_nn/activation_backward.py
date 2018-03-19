## Created by : Nilay

import numpy as np


def sigmoid_backward(dA, activation_cache):
    #   This function calculates the derivative dZ = dA*g'(z) (element-wise multi.)
    #   In case of sigmoid function it always comes out to be A[l] - Y or sigma(z[l]) - Y
    Z = activation_cache
    sig = 1/(1+np.exp(-Z))
    dZ = dA*sig*(1-sig)

    assert(dZ.shape == Z.shape)
    return dZ
    # return np.multiply(dA, np.multiply(sigmoid(Z), 1-sigmoid(Z)))     # I don't know why this didn't worked

def relu_backward(dA, activation_cache):
    Z = activation_cache

    dZ = np.array(dA, copy=True) #  Converting dZ to a correct object
    dZ[Z<=0] = 0                 #  convert all entries of dZ to 0 where Z<=0

    assert (dZ.shape == Z.shape)

    return dZ


def softmax_backward(dA, activation_cache):
    return dA
