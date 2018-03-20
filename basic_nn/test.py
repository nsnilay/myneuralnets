from myneuralnets import *
import myneuralnets
from test_case import *
from dnn_app_utils import *
import numpy as np
#
# parameters = myneuralnets.initialize_parameters([3,4,5,1])
# AL, caches = L_forward_propagation(np.array(([1,2,3],[2,3,4],[3,2,1])),parameters)
# print(AL)
# print(compute_cost(np.random.rand(1,5), np.random.rand(1,5)))
#
# X, parameters = L_model_forward_test_case_2hidden()
# AL, caches = L_forward_propagation(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))
#
# AL, Y_assess, caches = L_model_backward_test_case()
# grads = back_propagation(AL, Y_assess, caches)
# #print_grads(grads)
#
#
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)

# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

# layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
# parameters = classification_nn(train_x, train_y, layers_dims,learning_rate=0.01, optimizer='adam', num_iterations = 2500, print_cost = True)
#
# pred_train = predict(train_x, train_y, parameters)
#
# pred_test = predict(test_x, test_y, parameters)
