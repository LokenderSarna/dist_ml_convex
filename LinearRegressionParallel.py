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
        
        """
        # Gather data from train and test csv 
        data_train = pd.read_csv('../train.csv', skipinitialspace=1, index_col=0, parse_dates=True)

        # Explode datetime into hour, dayofweek, and month
        data_train['hour'] = data_train.index.hour
        data_train['dayofweek'] = data_train.index.weekday
        data_train['month'] = data_train.index.month

        # Build a dataset with its matching y outputs
        X_selected_cols = [u'weather',u'dayofweek',u'hour',u'season',u'holiday',u'workingday',u'temp',u'atemp',u'humidity',u'windspeed']
        X_train = data_train[X_selected_cols]
        y_train = data_train['count']
        y_train = np.matrix(y_train).T
        
        # Now we need to add a column of ones to account for the intercept in finding a linear prediction
        X_train = np.column_stack((X_train, np.ones(len(X_train))))
        
        # Split the training and testing data into 4 subsets, pretending we have 5 servers total, 4 of which are workers
        data_bcast = split_seq(X_train, 4) + split_seq(y_train, 4)        
        """
        
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
        data_bcast = split_seq(diabetes_X_train, 16) + split_seq(diabetes_y_train, 16)
                
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
    X6 = data_bcast[5]
    X7 = data_bcast[6]
    X8 = data_bcast[7]
    X9 = data_bcast[8]
    X10 = data_bcast[9]
    X11 = data_bcast[10]
    X12 = data_bcast[11]
    X13 = data_bcast[12]
    X14 = data_bcast[13]
    X15 = data_bcast[14]
    X16 = data_bcast[15]
        
    y1 = data_bcast[16]
    y2 = data_bcast[17]
    y3 = data_bcast[18]
    y4 = data_bcast[19]
    y5 = data_bcast[20]
    y6 = data_bcast[21]
    y7 = data_bcast[22]
    y8 = data_bcast[23]
    y9 = data_bcast[24]
    y10 = data_bcast[25]
    y11 = data_bcast[26]
    y12 = data_bcast[27]
    y13 = data_bcast[28]
    y14 = data_bcast[29]
    y15 = data_bcast[30]
    y16 = data_bcast[31]
    
    # Initializing our gradient function for the first order oracles
    grad_f1 = lambda x : 2*(np.dot(np.dot(X1.T, X1), x) - np.dot(X1.T, y1))
    grad_f2 = lambda x : 2*(np.dot(np.dot(X2.T, X2), x) - np.dot(X2.T, y2))
    grad_f3 = lambda x : 2*(np.dot(np.dot(X3.T, X3), x) - np.dot(X3.T, y3))
    grad_f4 = lambda x : 2*(np.dot(np.dot(X4.T, X4), x) - np.dot(X4.T, y4))
    grad_f5 = lambda x : 2*(np.dot(np.dot(X5.T, X5), x) - np.dot(X5.T, y5))
    grad_f6 = lambda x : 2*(np.dot(np.dot(X6.T, X6), x) - np.dot(X6.T, y6))
    grad_f7 = lambda x : 2*(np.dot(np.dot(X7.T, X7), x) - np.dot(X7.T, y7))
    grad_f8 = lambda x : 2*(np.dot(np.dot(X8.T, X8), x) - np.dot(X8.T, y8))
    
    grad_f9 = lambda x : 2*(np.dot(np.dot(X9.T, X9), x) - np.dot(X9.T, y9))
    grad_f10 = lambda x : 2*(np.dot(np.dot(X10.T, X10), x) - np.dot(X10.T, y10))
    grad_f11 = lambda x : 2*(np.dot(np.dot(X11.T, X11), x) - np.dot(X11.T, y11))
    grad_f12 = lambda x : 2*(np.dot(np.dot(X12.T, X12), x) - np.dot(X12.T, y12))
    grad_f13 = lambda x : 2*(np.dot(np.dot(X13.T, X13), x) - np.dot(X13.T, y13))
    grad_f14 = lambda x : 2*(np.dot(np.dot(X14.T, X14), x) - np.dot(X14.T, y14))
    grad_f15 = lambda x : 2*(np.dot(np.dot(X15.T, X15), x) - np.dot(X15.T, y15))
    grad_f16 = lambda x : 2*(np.dot(np.dot(X16.T, X16), x) - np.dot(X16.T, y16))
    
    # The dimensions for the w weight vector we are searching for is 2
    dim_f = 2
    
    # Now define the oracles for each server
    oracle1 = FirstOrderOracle(grad_f1, dim_f)
    oracle2 = FirstOrderOracle(grad_f2, dim_f) 
    oracle3 = FirstOrderOracle(grad_f3, dim_f) 
    oracle4 = FirstOrderOracle(grad_f4, dim_f)
    oracle5 = FirstOrderOracle(grad_f5, dim_f)
    oracle6 = FirstOrderOracle(grad_f6, dim_f) 
    oracle7 = FirstOrderOracle(grad_f7, dim_f) 
    oracle8 = FirstOrderOracle(grad_f8, dim_f)
    oracle9 = FirstOrderOracle(grad_f9, dim_f)
    oracle10 = FirstOrderOracle(grad_f10, dim_f) 
    oracle11 = FirstOrderOracle(grad_f11, dim_f) 
    oracle12 = FirstOrderOracle(grad_f12, dim_f)
    oracle13 = FirstOrderOracle(grad_f13, dim_f)
    oracle14 = FirstOrderOracle(grad_f14, dim_f) 
    oracle15 = FirstOrderOracle(grad_f15, dim_f) 
    oracle16 = FirstOrderOracle(grad_f16, dim_f)
    
    
    oracles = [oracle1, oracle2, oracle3, oracle4, oracle5, oracle6, oracle7, oracle8, oracle9, oracle10, oracle11, oracle12, oracle13, oracle14, oracle15, oracle16]
    
    # Testing for gossip implementation
    L = np.linalg.norm(np.dot(X2.T,X2))
    print L
    grad = GradientDescentGossip(oracles, max_iter=1000000, alpha=(lambda x : L))
    
    # Execute the gradient descent and attempt to find a weight w
    # grad = GradientDescent(oracles, lipschitz=True, max_iter=1000000, epsilon=0.5)
    
    # max_iter = 10000
    # alpha = (lambda iteration : 0.001 if iteration < 0.9*max_iter else 0.00001 )
    # grad = GradientDescent(oracles, alpha=alpha, max_iter=max_iter, epsilon=0.001)
    
    # max_iter = 1000000
    # alpha = (lambda iteration : 0.0000000168 if iteration < 0.8*max_iter else 0.000000001 )
    # grad = GradientDescent(oracles, alpha=alpha, max_iter=max_iter, epsilon=0.1)
    
    
    # t0 = time.time()
    grad.execute()
    # t1 = time.time()
    # if rank == 0:
    #     print "Execution time = %f" %(t1-t0)
    
        
# Source: http://code.activestate.com/recipes/425397-split-a-list-into-roughly-equal-sized-pieces/
def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq
    
if __name__ == "__main__":
    main()