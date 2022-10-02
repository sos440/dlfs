from numba import jit, njit

import pickle
import numpy as np
import time
import math

from asyncio.windows_events import NULL
import sys, os
from dataset.mnist import load_mnist
from PIL import Image

sys.path.append(os.pardir)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# Import data
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten = True, normalize = True, one_hot_label = False)


# Mode
mode_continue = True
mode_learn = True
# sigmoid / relu / smooth_relu
mode_activation_fn = 'sigmoid'


# Hyperparameters:

# Calculate step size based on the size of the gradient
def f_step_size(grad_size):
    # 1.0 / (1.0 + 2.0 * max(0, -math.log10(grad_size)))
    return 1.0 + 2.0 * max(0, -math.log10(grad_size))

# Specify the hidden layers via the number of nodes
hidden_layers = [16, 16]

# Threshold for gradient descent
threshold_GD = 0.00000001


# Basic deep neural network, optimized for MNIST
# Do not change anything below:

num_train = len(x_train)
num_test = len(x_test)

# Various activation-related functions
f_list = {}

# sigmoid
@njit
def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))

@njit
def f_sigmoid_prime(x):
    return 1 / ((1 + np.exp(x)) * (1 + np.exp(-x)))

f_list['sigmoid'] = (f_sigmoid, f_sigmoid_prime)

# relu
@njit
def f_relu(x):
    return max(0, x)

@njit
def f_relu_prime(x):
    if (x > 0):
        return 1
    else:
        return 0

f_list['relu'] = (f_relu, f_relu_prime)

# smooth relu
__smooth_relu_param = 2.0
@njit
def f_smooth_relu(x):
    return 0.5 * (x + math.sqrt(__smooth_relu_param + x*x))

@njit
def f_smooth_relu_prime(x):
    return 0.5 * (1 + x / math.sqrt(__smooth_relu_param + x*x))

f_list['smooth_relu'] = (f_smooth_relu, f_smooth_relu_prime)

# softmax
@njit
def f_softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Vectorized activation functions
f_act, f_act_prime = f_list[mode_activation_fn]
vf_act = np.vectorize(f_act)
vf_act_prime = np.vectorize(f_act_prime)
vf_softmax = np.vectorize(f_softmax, signature='(n)->(m)')
print(f'Activation function chosen as: {mode_activation_fn}')

# Define the augmented specification
layers = [len(x_train[0])] + hidden_layers + [10]
num_layers = len(layers)

# Index of the last layer
# This is because the last layer uses the softmax for activation instead of the sigmoid.
idx_layer_last = num_layers - 1

# Standard basis vectors for the output layer
ell = np.eye(10)

# Initialize weight/bias pair
def wb_pairs(f):
    weights = [0] * num_layers
    bias = [0] * num_layers

    for i in range(1, num_layers):
        weights[i] = f((layers[i-1], layers[i]))
        bias[i] = f(layers[i])

    return weights, bias    

param_W, param_b = wb_pairs(np.random.standard_normal)

# Save the network
def save_network():
    with open('muy_network.pkl', 'wb') as f:
        pickle.dump([layers, param_W, param_b], f)

def load_network():
    with open('muy_network.pkl', 'rb') as f:
        return pickle.load(f)

# Test the performance
def test(x_list, t_list):
    a_list = x_list
    for idx in range(1, num_layers):
        z_list = np.dot(a_list, param_W[idx]) + param_b[idx]
        if (idx < idx_layer_last):
            a_list = vf_act(z_list)
        else:
            a_list = vf_softmax(z_list)

    return np.count_nonzero(np.argmax(a_list, axis=1) == t_list), np.size(t_list)

# Learning via gradient descent
def learn(max_lessons):
    for lesson in range(max_lessons):
        # Debug
        time_0 = time.time()
        print(f'Step {lesson + 1}:')
        
        # Compute node values
        z_lists = [None] * num_layers
        a_lists = [None] * num_layers
        a_lists[0] = x_train
        for idx in range(1, num_layers):
            z_lists[idx] = np.dot(a_lists[idx-1], param_W[idx]) + param_b[idx]
            if (idx < idx_layer_last):
                a_lists[idx] = vf_act(z_lists[idx])
            else:
                a_lists[idx] = vf_softmax(z_lists[idx])

        # Compute the gradient of the total squared error (TSE) by backpropagation
        dTSE_dW, dTSE_db = wb_pairs(np.zeros)
        for i in range(num_train):
            dSE_db = [0] * num_layers
            for k in range(idx_layer_last, 0, -1):
                if (k == idx_layer_last):
                    m = a_lists[k][i]
                    t = t_train[i]
                    dSE_db[k] = 2 * (-np.dot(m - ell[t], m) * m + np.square(m) - m[t] * ell[t])
                else:
                    dSE_db[k] = np.multiply(vf_act_prime(z_lists[k][i]), np.dot(param_W[k+1], dSE_db[k+1]))
                
                dTSE_db[k] += dSE_db[k]
                dTSE_dW[k] += np.outer(a_lists[k-1][i], dSE_db[k])
        
        # Debug: Report training performance
        num_correct, num_total = np.count_nonzero(np.argmax(a_lists[idx_layer_last], axis=1) == t_train), np.size(t_train)
        print(f'    Training performance: {100 * num_correct / num_total:.4f}%')
        
        # Compute the norm of the gradient of MSE
        dMSE_size = 0
        for idx in range(1, num_layers):
            dMSE_size += np.linalg.norm(dTSE_dW[idx]) / num_train
            dMSE_size += np.linalg.norm(dTSE_db[idx]) / num_train
        
        # Update the parameters W and b
        eps = f_step_size(dMSE_size)
        print(f'    Step size: {eps:.4f}')
        for idx in range(1, num_layers):
            param_W[idx] -= (eps / num_train) * dTSE_dW[idx]
            param_b[idx] -= (eps / num_train) * dTSE_db[idx]

        # Debug
        time_1 = time.time()
        print(f'    Norm of the gradient: {dMSE_size}')
        print(f'    Time elapsed: {time_1 - time_0:.2f} s')
        
        save_network()
        print('    Network saved.')

        # If the gradient is small
        if dMSE_size < threshold_GD:
            break

        # Debug: For each 20 lessons, report mid-learning performance
        if ((lesson + 1) % 20 == 0):
            num_correct, num_total = test(x_test, t_test)
            print(f'\nMid-learning performance: {num_correct}/{num_total} ({100 * num_correct / num_total:.4f}%)\n')
            


# Flow control

if mode_continue:
    t_layers, t_param_W, t_param_b = load_network()
    if np.array_equal(layers, t_layers):
        print('Continuing from the save.\n')
        layers, param_W, param_b = t_layers, t_param_W, t_param_b
else:
    print('Starting a new learning.\n')

if mode_learn:
    learn(10000)

# Report the final performance
num_correct, num_total = test(x_test, t_test)
print(f'Final performance: {num_correct}/{num_total} ({100 * num_correct / num_total:.4f}%)')