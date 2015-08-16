from dist_ml_convex import *
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import time

# Source: http://code.activestate.com/recipes/425397-split-a-list-into-roughly-equal-sized-pieces/
def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq

# So here we are using sklearns simple diabetes example where we will use just one feature to predict y values
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis]    
diabetes_X_temp = diabetes_X[:, :, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-20]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_train = np.matrix(diabetes_y_train).T

# Now we need to add a column of ones to account for the intercept in finding a linear prediction
diabetes_X_train = np.column_stack((diabetes_X_train, np.ones(len(diabetes_X_train))))

# Split the training and testing data into 4 subsets, pretending we have 5 servers total, 4 of which are workers
Xs = split_seq(diabetes_X_train, 2)
ys = split_seq(diabetes_y_train, 2)
X0, X1, y0, y1 = Xs[0], Xs[1], ys[0], ys[1]

# Initializing our gradient function for the first order oracles
grad_f0 = lambda x : 2*(np.dot(np.dot(X0.T, X0), x) - np.dot(X0.T, y0))
grad_f1 = lambda x : 2*(np.dot(np.dot(X1.T, X1), x) - np.dot(X1.T, y1))
#grad_f2 = lambda x : 2*(np.dot(np.dot(X2.T, X2), x) - np.dot(X2.T, y2))


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

x = np.zeros((2,1))

cont_iter = True
num_iter = 0

halfs = np.zeros((2,1))
halfs.fill(0.5)
alpha = np.zeros((2,1))
alpha.fill(0.001)

# Using Graph: 0 - 1 - 2
while num_iter < 100000:
    
    num_iter += 1
    
    grad_recv = np.zeros((2,1))

    if rank == 0:
        comm.Send(grad_f0(x), dest=1)
        comm.Recv(grad_recv, source=1)
        #x = x - alpha * grad_recv
        x = x - alpha * np.asarray(grad_f0(x))
    
    if rank == 1:
        comm.Send(grad_f1(x), dest=0)
        comm.Recv(grad_recv, source=0)
        #x = x - alpha * grad_recv
        x = x - alpha * np.asarray(grad_f1(x))

    print x

