from numba import jit, njit

import pickle
import numpy as np
import time

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


# Hyperparameters:

# Step size
step_size = 7.0

# Specify the hidden layers via the number of nodes
hidden_layers = [16, 16]

# Threshold for gradient descent
threshold_GD = 0.00000001


# Basic deep neural network, optimized for MNIST
# Do not change anything below:

# a(x) : Activation function
@njit
def f_act(x):
    return 1 / (1 + np.exp(-x))

# a'(x)
@njit
def f_act_prime(x):
    return 1 / ((1 + np.exp(x)) * (1 + np.exp(-x)))

# Softmax function
@njit
def f_softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Vectorized activation functions
vf_act = np.vectorize(f_act)
vf_act_prime = np.vectorize(f_act_prime)
vf_softmax = np.vectorize(f_softmax, signature='(n)->(m)')

# Create parameters
@njit
def create_Wb_pair(layers):
    num_layers = np.shape(layers)[0]
    param_W = [0] * num_layers
    param_b = [0] * num_layers
    for i in range(1, num_layers):
        param_W[i] = np.zeros((layers[i-1], layers[i]))
        param_b[i] = np.zeros(layers[i])

    return param_W, param_b

# Define the augmented specification
layers = [np.shape(x_train[0])[0]] + hidden_layers + [10]

# Initialize weight/bias pair
param_W, param_b = create_Wb_pair(layers)

# Save the network
def save_network(layers, param_W, param_b):
    with open('muy_network.pkl', 'wb') as f:
        pickle.dump([layers, param_W, param_b], f)

def load_network():
    with open('muy_network.pkl', 'rb') as f:
        return pickle.load(f)

# Compute performance
# a_list : list of probability vectors
# t_list : list of correct answers
# returns num_correct, num_total
@njit
def compute_performance(a_list, t_list):
    return np.count_nonzero(np.argmax(a_list, axis=1) == t_list), np.shape(t_list)[0]

# Compute output
@njit
def compute_output(layers, param_W, param_b, x_list):
    num_layers = np.shape(layers)[0]
    idx_output_layer = num_layers - 1

    z_lists = [0] * num_layers
    a_lists = [0] * num_layers
    a_lists[0] = x_list
    for idx in range(1, num_layers):
        z_lists[idx] = np.dot(a_lists[idx-1], param_W[idx]) + param_b[idx]
        if (idx < idx_output_layer):
            a_lists[idx] = vf_act(z_lists[idx])
        else:
            a_lists[idx] = vf_softmax(z_lists[idx])
    
    return z_lists, a_lists

# Compute the gradient of TSE
@njit
def compute_grad(layers, param_W, param_b, x_list, t_list):
    num_layers = np.size(layers)
    num_examples = np.shape(x_list)[0]
    idx_output_layer = num_layers - 1

    # Standard basis vectors for the output layer
    ell = np.eye(10)

    # Compute node values
    z_lists, a_lists = compute_output(layers, param_W, param_b, x_list)

    # Compute the gradient of the total squared error (TSE) by backpropagation
    dTSE_dW, dTSE_db = create_Wb_pair(layers)
    for i in range(num_examples):
        dSE_db = [0] * num_layers
        for k in range(idx_output_layer, 0, -1):
            if (k == idx_output_layer):
                m = a_lists[k][i]
                t = t_list[i]
                dSE_db[k] = 2 * (-np.dot(m - ell[t], m) * m + np.square(m) - m[t] * ell[t])
            else:
                dSE_db[k] = np.multiply(vf_act_prime(z_lists[k][i]), np.dot(param_W[k+1], dSE_db[k+1]))
            
            dTSE_db[k] += dSE_db[k]
            dTSE_dW[k] += np.outer(a_lists[k-1][i], dSE_db[k])
    
    # Compute the norm of the gradient of MSE
    dMSE_size = 0
    for idx in range(1, num_layers):
        dMSE_size += np.linalg.norm(dTSE_dW[idx]) / num_examples
        dMSE_size += np.linalg.norm(dTSE_db[idx]) / num_examples
    
    return dTSE_dW, dTSE_db, dMSE_size, a_lists[idx_output_layer]

# Learning via gradient descent
def learn(max_lessons):
    num_layers = np.size(layers)
    num_train = np.shape(x_train)[0]
    idx_output_layer = num_layers - 1

    for lesson in range(max_lessons):
        # Debug
        time_0 = time.time()
        print(f'Step {lesson + 1}:')
        
        # Compute the gradient of TSE
        dTSE_dW, dTSE_db, dMSE_size, a_list = compute_grad(layers, param_W, param_b, x_train, t_train)
        
        # Debug: Report training performance
        num_correct, num_total = compute_performance(a_list, t_train)
        print(f'    Training performance: {100 * num_correct / num_total:.4f}%')
        
        # Update the parameters W and b
        for idx in range(1, num_layers):
            param_W[idx] -= (step_size / num_train) * dTSE_dW[idx]
            param_b[idx] -= (step_size / num_train) * dTSE_db[idx]

        # Debug
        time_1 = time.time()
        print(f'    Norm of the gradient: {dMSE_size}')
        print(f'    Time elapsed: {time_1 - time_0:.2f} s')
        
        save_network(layers, param_W, param_b)
        print('    Network saved.')

        # If the gradient is small
        if dMSE_size < threshold_GD:
            break

        # Debug: For each 20 lessons, report mid-learning performance
        if ((lesson + 1) % 20 == 0):
            num_correct, num_total = compute_performance(compute_output(layers, param_W, param_b, x_test)[1, idx_output_layer], t_test)
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
    learn(1000)

# Report the final performance
num_correct, num_total = compute_performance(compute_output(layers, param_W, param_b, x_test)[1, np.size(layers) - 1], t_test)
print(f'Final performance: {num_correct}/{num_total} ({100 * num_correct / num_total:.4f}%)')