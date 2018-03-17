## Created by : Nilay

# This Neural Nets module is for L-layer logistic classification problem. This is a vectorized implementation
# so that no loop is used and thus computation is faster. The input data is a numpy array
# Also, the L-1 layers uses RELU activation and the Lth layer uses SIGMOID activation

# How to use - Step 1) import this module
#              Step 2) use classification_nn(X, Y, layers_dims, num_iterations)

#   While I was develpoing this and testing,I found out that the hyperparameters that I used played a crucial role
#   to improve costs. If you'll tweak the initialization or learning rate you might see completely different results.

import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(layer_dims):
    #Input : python array list containing the dimension of each layer of the network
    #Output : python dictionary containing those initialize_parameters

    #   Sometimes parameter initialization plays an important role. If your cost isn't
    #   improving, you can also try different initialization, but always try to use small
    #   initialization.

    parameters = {}
    L = len(layer_dims)

    # Loop through 1 to L-1 (both inclusive)
    for i in range(1,L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])/ np.sqrt(layer_dims[i-1])# * 0.01      #randn initiates a 'std. normal' distribution,
                        #division by 100 ensure small weights to avoid runtime overflow errors. This one works good for RELU activation
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))                            #initializing bias with zeros
        assert(parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i-1]))
        assert(parameters['b' + str(1)].shape == (layer_dims[1], 1))

    return parameters

def one_step_forward(A_prev, W, b, activation):

    Z = np.dot(W,A_prev) + b
    linear_cache = A_prev, W, b

    if activation == 'relu':                        # This relu function is for L-1 layers
        A, activation_cache = relu(Z)

    if activation == 'sigmoid':                     # The sigmoid function is for last Lth layer call
        A, activation_cache = sigmoid(Z)            # Remember the dimension of this layer is 1

    return A, (linear_cache, activation_cache)

def L_forward_propagation(X, parameters):
    #   Input : 'X' is the input numpy array (input size, no. of examples), parameters contain weights and bias
    #   Output : list of caches containing every step of forward propagation

        caches = []
        A = X
        L = len(parameters) // 2        # no. of layers in the network, '//' is used in Python 3 for integer division

        for i in range(1,L):
            A_prev = A
            A, cache = one_step_forward(A_prev, parameters['W' + str(i)],   # for L-1 layer we'll call relu activation layer
                                            parameters['b' + str(i)],
                                            activation = 'relu')
            caches.append(cache)

        AL, cache = one_step_forward(A, parameters['W' + str(L)],
                                        parameters['b' + str(L)],
                                        activation = 'sigmoid')
        caches.append(cache)

        assert(AL.shape == (1, X.shape[1]))
        return AL, caches

def sigmoid(Z):
    return (1/(1+np.exp(-(Z)))), Z

def relu(Z):
    return np.maximum(0,Z), Z

def compute_cost(AL, Y):
    # Input : probability vector of the classification (1, no. of examples)
    # Output : The method computes the cross-entropy cost

    m = Y.shape[1]
    cost = (1/m) * np.sum(-np.multiply(Y, np.log(AL)) - np.multiply(1-Y, np.log(1-AL)))  # np.multiply for element-wise multiplication
    cost = np.squeeze(cost)                 # Remove single-dimensional entries from the shape of an array.
                                            # >>> x = np.array([[[0], [1], [2]]])
                                            # >>> x.shape
                                            # (1, 3, 1)
                                            # >>> np.squeeze(x).shape
                                            # (3,)
                                            # >>> np.squeeze(x, axis=0).shape
                                            # (3, 1)
    assert(cost.shape == ())
    return cost

def one_step_backward(dA, cache, activation):
    linear_cache, activation_cache = cache      #   one entry of cache = ((Al, Wl, bl), Zl)
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]                         #   m is no. of examples

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)

    #   The three outputs(dW[l], db[l], dA[l]) are computed using the following equations -
    #   dW[l] = dLoss/dW[l] = 1/m * dZ[l]A[l-1].T
    #   db[l] = dLoss/db[l] = 1/m * summation over i=1 t0 m in dZ[l][i]
    #   dA[l-1] = dLoss/dA[l-1] = W[l].TdZ[l]

    dW = np.dot(dZ, A_prev.T) / m
    db = (np.sum(dZ, axis = 1, keepdims = True)) / m
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

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

def back_propagation(AL, Y, caches):
    #   Input : AL - probab vector, O/P of the L_forward_step
    #           Y - true "label" vector (0 or 1)
    #           caches -
    #   Output : grads - a dictionary with gradients

    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)              # number of layers

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  #    dLoss/dAL

    #   Grads of last "Lth" layer that will be SIGMOID layer
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = one_step_backward(dAL, current_cache, 'sigmoid')

    #    loop from L-2 to 0 that will be RELU layer
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_step_backward(grads["dA" + str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

def classification_nn(X, Y, layer_dims, learning_rate = 0.005, num_iterations = 3000, print_cost = True):
    #   Implements an L-layer neural network [LINEAR=>RELU]*(L-1) -> [LLINEAR->SIGMOID]
    #   Input : X - numpy I/P array (no. of examples, l*b*3)
    #           Y - true label vector (0 or 1) (1, no. of examples)
    #           layer_dims - a list containing dims. of each layer with last dim as 1
    #   Output : parameters learned during training

    costs = []

    parameters = initialize_parameters(layer_dims)      # initialise Weights and biases with random and zeros resp.

    for i in range(0, num_iterations):
        # Steps involving training neural nets
        AL, caches = L_forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = back_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

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
