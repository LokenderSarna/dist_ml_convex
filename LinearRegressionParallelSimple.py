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
        
        # So here we are using sklearns simple diabetes example where we will use just one feature to predict y values
        diabetes = datasets.load_diabetes()
    
        # Use only one feature
        diabetes_X = diabetes.data[:, np.newaxis]    
        diabetes_X_temp = diabetes_X[:, :, 2]
    
        # Split the data into training/testing sets
        diabetes_X_train = diabetes_X_temp[:-20]
        #diabetes_X_test = diabetes_X_temp[-20:]

        # Split the targets into training/testing sets
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_train = np.matrix(diabetes_y_train).T
        #diabetes_y_test = diabetes.target[-20:]
    
        # Now we need to add a column of ones to account for the intercept in finding a linear prediction
        #diabetes_X_train = np.matrix(diabetes_X_train)
        diabetes_X_train = np.column_stack((diabetes_X_train, np.ones(len(diabetes_X_train))))
    
        # Split the training and testing data into 4 subsets, pretending we have 5 servers total, 4 of which are workers
        data_bcast = split_seq(diabetes_X_train, 5) + split_seq(diabetes_y_train, 5)
                
    # The other processes are receiving the data so they send None
    else:
        data_bcast = None
    
    # Here all processes receive the broadcasted data, this saves us doing all the data splitting in each process
    data_bcast = comm.bcast(data_bcast, root=0)
    
    # Now we declare each of the split data sets to send to our parallel gradient descent execution
    X1 = data_bcast[0]
    X2 = data_bcast[1]
    X3 = data_bcast[2]
    X4 = data_bcast[3]
    X5 = data_bcast[4]
        
    y1 = data_bcast[5]
    y2 = data_bcast[6]
    y3 = data_bcast[7]
    y4 = data_bcast[8]
    y5 = data_bcast[9]
    
    # Initializing our gradient function for the first order oracles
    grad_f1 = lambda x : 2*(np.dot(np.dot(X1.T, X1), x) - np.dot(X1.T, y1))
    grad_f2 = lambda x : 2*(np.dot(np.dot(X2.T, X2), x) - np.dot(X2.T, y2))
    grad_f3 = lambda x : 2*(np.dot(np.dot(X3.T, X3), x) - np.dot(X3.T, y3))
    grad_f4 = lambda x : 2*(np.dot(np.dot(X4.T, X4), x) - np.dot(X4.T, y4))
    grad_f5 = lambda x : 2*(np.dot(np.dot(X5.T, X5), x) - np.dot(X5.T, y5))
    
    # The dimensions for the w weight vector we are searching for is 2
    dim_f = 2
    
    # Now define the oracles for each server
    oracle1 = FirstOrderOracle(grad_f1, dim_f)
    oracle2 = FirstOrderOracle(grad_f2, dim_f)
    oracle3 = FirstOrderOracle(grad_f3, dim_f)
    oracle4 = FirstOrderOracle(grad_f4, dim_f)
    oracle5 = FirstOrderOracle(grad_f5, dim_f)
    
    oracles = [oracle1, oracle2, oracle3, oracle4, oracle5]
    
    # Testing for gossip implementation
    x_init = np.zeros((2,1))
    grad = GradientDescentGossip(oracles, max_iter=100000, alpha=lambda x: 0.003)   
    
    grad.execute()
    
        
# Source: http://code.activestate.com/recipes/425397-split-a-list-into-roughly-equal-sized-pieces/
def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq
    
if __name__ == "__main__":
    main()