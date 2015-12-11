from dist_ml_convex import *
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import time

def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
        
    # If the process is the master assign subsets of the dataset to each process via oracles
    if rank == 0:
        
        X_train, y_train = load_diabetes()
        # X_train, y_train = load_random()
    
        # Split the training and testing data into n sets where n=number of workers (#processes - 1)
        data_bcast = split_seq(X_train, size-1) + split_seq(y_train, size-1)
                
    # The other processes are receiving the data so they send None
    else:
        data_bcast = None
    
    # Here all processes receive the broadcasted data, this saves us doing all the data splitting in each process
    data_bcast = comm.bcast(data_bcast, root=0)
 
    # Now we declare each of the split data sets to send to our parallel gradient descent execution
    oracles = []
    for i in range(size-1):
        X_train_data = data_bcast[i]
        y_train_data = data_bcast[i+size-1]
        grad_f = lambda x : 2*(np.dot(np.dot(X_train_data.T, X_train_data), x) - np.dot(X_train_data.T, y_train_data))
        dim_f = X_train_data.shape[1]
        oracle = FirstOrderOracle(grad_f, dim_f)
        oracles.append(oracle)
 
    # Testing for master-worker implementation
    x_init = np.zeros((dim_f,1))
    if rank == 0: start_time = time.time()
    grad = GradientDescent(oracles, max_iter=1000000, x_init=x_init, alpha=lambda x: 0.002)   
    grad.execute()
    if rank == 0: print("--Done: %s seconds --" % (time.time() - start_time))
        
# Source: http://code.activestate.com/recipes/425397-split-a-list-into-roughly-equal-sized-pieces/
def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq
        
def load_diabetes():
    # load dataset
    dataset = datasets.load_diabetes()

    # Split the data into training/testing sets
    X = dataset.data[:, np.newaxis]
    X_train = X[:-20]
    # Now we need to add a column of ones to account for the intercept in finding a linear prediction
    X_train = np.matrix(X_train)
    X_train = np.column_stack((X_train, np.ones(len(X_train))))

    # Split the targets into training/testing sets
    y_train = dataset.target[:-20]
    y_train = np.matrix(y_train).T
    # print type(dataset.target)

    return X_train, y_train
    
    
# def load_bikesharing():
    # copy form TestSingle.py
    
    
def load_random():
    X_train = np.random.rand(10000,1)
    X_train = np.matrix(X_train)
    # X_train = np.column_stack((X_train, np.ones(len(X_train))))
    
    y_train = np.random.rand(1,10000)
    y_train = np.matrix(y_train).T

    return X_train, y_train
    
if __name__ == "__main__":
    main()