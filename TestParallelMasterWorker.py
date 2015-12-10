from dist_ml_convex import *
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import time

def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    dim_f = 11
        
    # If the process is the master assign subsets of the dataset to each process via oracles
    if rank == 0:
        
        X_train, y_train = load_dataset(dim_f)
    
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
        oracle = FirstOrderOracle(grad_f, dim_f)
        oracles.append(oracle)
 
    # Testing for master-worker implementation
    x_init = np.zeros((dim_f,1))
    if rank == 0: start_time = time.time()
    grad = GradientDescent(oracles, max_iter=1000000, x_init=x_init, alpha=lambda x: 0.001)   
    grad.execute()
    if rank == 0: print("--Done: %s seconds --" % (time.time() - start_time))
        
# Source: http://code.activestate.com/recipes/425397-split-a-list-into-roughly-equal-sized-pieces/
def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq
        
def load_dataset(dim):
    # So here we are using sklearns simple diabetes example where we will use just one feature to predict y values
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis]  
    diabetes_X_temp = diabetes_X[:, :, :(dim-1)]
    # print diabetes_X_temp.shape
    
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X_temp[:-20]  
    #diabetes_X_test = diabetes_X_temp[-20:]
    # print diabetes_X_train.shape

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_train = np.matrix(diabetes_y_train).T
    #diabetes_y_test = diabetes.target[-20:]
    # print diabetes_y_train.shape
    
    # Now we need to add a column of ones to account for the intercept in finding a linear prediction
    diabetes_X_train = np.matrix(diabetes_X_train)
    diabetes_X_train = np.column_stack((diabetes_X_train, np.ones(len(diabetes_X_train))))
    # print diabetes_X_train.shape
    
    return diabetes_X_train, diabetes_y_train
    
if __name__ == "__main__":
    main()