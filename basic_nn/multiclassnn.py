##  Created by : Nilay
# Comments have to be added and some numerical stability issues have to be resolved


import numpy as np
import optimizers
import matplotlib.pyplot as plt
from myneuralnets import initialize_parameters, update_parameters
from activations import relu, softmax
from activation_backward import relu_backward, softmax_backward

def one_step_forward(A_prev, W, b, activation):

    Z = np.dot(W, A_prev) + b
    linear_cache = A_prev, W, b
    # print(A_prev.shape,W.shape,b.shape)

    if activation == 'relu':
        A, activation_cache = relu(Z)
    if activation == 'softmax':
        A, activation_cache = softmax(Z)

    return A, (linear_cache, activation_cache)

def L_forward_propagation(X, parameters):

    L = len(parameters) // 2
    caches = []
    A = X

    for l in range(1,L):
        A_prev = A
        A, cache = one_step_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],
                                                        activation = 'relu')
        caches.append(cache)

    AL, cache = one_step_forward(A, parameters["W"+str(L)], parameters["b"+str(L)],
                                                        activation = 'softmax')
    caches.append(cache)

    # assert(AL.shape == (parameters["b"+str(L-1)].shape, X.shape[1]))
    return AL, caches

def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y, np.log(AL))) / m
    cost = np.squeeze(cost)

    assert(cost.shape == ())
    return cost

def one_step_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    if activation == 'softmax':
        dZ = softmax_backward(dA, activation_cache)

    dW = np.dot(dZ, A_prev.T) / m
    db = (np.sum(dZ, axis = 1, keepdims = True)) / m
    dA_prev = np.dot(W.T, dZ)

    # print(dA_prev.shape,A_prev.shape)
    # print(dW.shape,W.shape)
    # print(db.shape,b.shape)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

def back_propagation(AL, Y, caches):
    #   Input : AL - probab vector, O/P of the L_forward_step
    #           Y - true "class" vector (0 or 1)
    #           caches -
    #   Output : grads - a dictionary with gradients


    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL =  AL - Y      ###- np.divide(Y,AL)
    # print("dAL shape", dAL.shape)

    current_cache = caches[L-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = one_step_backward(dAL,
                                                                current_cache, activation='softmax')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_step_backward(grads["dA"+str(l+1)],
                                                                current_cache, activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def multiclass_nn(X, Y, layer_dims, learning_rate = 0.005, optimizer = 'gd', num_iterations=3000, print_cost=True):

    costs = []

    parameters = initialize_parameters(layer_dims)

    for i in range(0, num_iterations):
        # Steps involving training neural nets
        AL, caches = L_forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = back_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate, optimizer)

        # printing cost after every 100th iteration
        if print_cost and i % 100 == 0:
            print("Cost after %i : %f" %(i,cost))
        if i % 10 == 0:
            costs.append(cost)

    # Plotting the costs
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
