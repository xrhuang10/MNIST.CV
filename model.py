#Coding a Neural Network from Scratch
#MNIST Digit Classification
#2 layers

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataframe = pd.read_csv("Dataset/mnist_train.csv")
dataframe.iloc[:, 1:] = (dataframe.iloc[:, 1:] != 0).astype(int)

data = np.array(dataframe)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[:1000].T
Y_test = data_dev[0]
X_test = data_dev[1: n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1: n]

def initialize_parameters():
    w1 = np.random.randn(100, 784)
    b1 = np.random.randn(10, 1)
    w2 = np.random.randn(10, 784)
    b2 = np.random.randn(10, 1)
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    return np.exp(z) / sum(np.exp(z))

def forward_propagate(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

def one_hot_encode(y):
    arr = np.zeros(y.size, y.max() + 1, dtype = int)
    arr[np.arange(y.size), y] = 1
    return arr.T

def dReLU(z):
    return z > 0

def backward_propagate(x, y, w2, z1, z2, a1, a2):
    m = y.size
    encoded_y = one_hot_encode(y)
    dz2 = a2 - encoded_y
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m * np.sum(dz2, 2)
    dz1 = w2.T.dot(dz2) * dReLU(a1)
    dw1 = 1/m * dz1.dot(x.T)
    db1 = 1/m * np.sum(dz1, 2)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def gradient_descent(x, y, iterations):
    w1, b1, w2, b2 = initialize_parameters()
    for i in range(iterations):
        a1, a2, z1, z2 = forward_propagate(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_propagate(x, y, w2, z1, z2, a1, a2)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
