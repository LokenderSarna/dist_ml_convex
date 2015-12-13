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
        
        # X_train, y_train = load_diabetes()
        # X_train, y_train = load_bikesharing()
        X_train, y_train = load_twitter()
        # X_train, y_train = load_powerconsumption()
    
        # Split the training and testing data into n sets where n=number of workers (#processes - 1)
        data_bcast = split_seq(X_train, size-1) + split_seq(y_train, size-1)
                
    # The other processes are receiving the data so they send None
    else:
        data_bcast = None
    
    # Here all processes receive the broadcasted data, this saves us doing all the data splitting in each process
    data_bcast = comm.bcast(data_bcast, root=0)
 
    # Now we declare each of the split data sets to send to our parallel gradient descent execution

    # X_train_1 = data_bcast[0]
    # y_train_1 = data_bcast[2]
    # X_train_2 = data_bcast[1]
    # y_train_2 = data_bcast[3]
    
    # X_train_1 = data_bcast[0]
    # y_train_1 = data_bcast[4]
    # X_train_2 = data_bcast[1]
    # y_train_2 = data_bcast[5]
    # X_train_3 = data_bcast[2]
    # y_train_3 = data_bcast[6]
    # X_train_4 = data_bcast[3]
    # y_train_4 = data_bcast[7]
    
    # X_train_1 = data_bcast[0]
    # y_train_1 = data_bcast[8]
    # X_train_2 = data_bcast[1]
    # y_train_2 = data_bcast[9]
    # X_train_3 = data_bcast[2]
    # y_train_3 = data_bcast[10]
    # X_train_4 = data_bcast[3]
    # y_train_4 = data_bcast[11]
    # X_train_5 = data_bcast[4]
    # y_train_5 = data_bcast[12]
    # X_train_6 = data_bcast[5]
    # y_train_6 = data_bcast[13]
    # X_train_7 = data_bcast[6]
    # y_train_7 = data_bcast[14]
    # X_train_8 = data_bcast[7]
    # y_train_8 = data_bcast[15]

    X_train_1 = data_bcast[0]
    y_train_1 = data_bcast[16]
    X_train_2 = data_bcast[1]
    y_train_2 = data_bcast[17]
    X_train_3 = data_bcast[2]
    y_train_3 = data_bcast[18]
    X_train_4 = data_bcast[3]
    y_train_4 = data_bcast[19]
    X_train_5 = data_bcast[4]
    y_train_5 = data_bcast[20]
    X_train_6 = data_bcast[5]
    y_train_6 = data_bcast[21]
    X_train_7 = data_bcast[6]
    y_train_7 = data_bcast[22]
    X_train_8 = data_bcast[7]
    y_train_8 = data_bcast[23]
    X_train_9 = data_bcast[8]
    y_train_9 = data_bcast[24]
    X_train_10 = data_bcast[9]
    y_train_10 = data_bcast[25]
    X_train_11 = data_bcast[10]
    y_train_11 = data_bcast[26]
    X_train_12 = data_bcast[11]
    y_train_12 = data_bcast[27]
    X_train_13 = data_bcast[12]
    y_train_13 = data_bcast[28]
    X_train_14 = data_bcast[13]
    y_train_14 = data_bcast[29]
    X_train_15 = data_bcast[14]
    y_train_15 = data_bcast[30]
    X_train_16 = data_bcast[15]
    y_train_16 = data_bcast[31]
    
    dim_f = X_train_1.shape[1]
    
    grad_f_1 = lambda x : 2*(np.dot(np.dot(X_train_1.T, X_train_1), x) - np.dot(X_train_1.T, y_train_1))
    grad_f_2 = lambda x : 2*(np.dot(np.dot(X_train_2.T, X_train_2), x) - np.dot(X_train_2.T, y_train_2))
    grad_f_3 = lambda x : 2*(np.dot(np.dot(X_train_3.T, X_train_3), x) - np.dot(X_train_3.T, y_train_3))
    grad_f_4 = lambda x : 2*(np.dot(np.dot(X_train_4.T, X_train_4), x) - np.dot(X_train_4.T, y_train_4))
    grad_f_5 = lambda x : 2*(np.dot(np.dot(X_train_5.T, X_train_5), x) - np.dot(X_train_5.T, y_train_5))
    grad_f_6 = lambda x : 2*(np.dot(np.dot(X_train_6.T, X_train_6), x) - np.dot(X_train_6.T, y_train_6))
    grad_f_7 = lambda x : 2*(np.dot(np.dot(X_train_7.T, X_train_7), x) - np.dot(X_train_7.T, y_train_7))
    grad_f_8 = lambda x : 2*(np.dot(np.dot(X_train_8.T, X_train_8), x) - np.dot(X_train_8.T, y_train_8))
    grad_f_9 = lambda x : 2*(np.dot(np.dot(X_train_9.T, X_train_9), x) - np.dot(X_train_9.T, y_train_9))
    grad_f_10 = lambda x : 2*(np.dot(np.dot(X_train_10.T, X_train_10), x) - np.dot(X_train_10.T, y_train_10))
    grad_f_11 = lambda x : 2*(np.dot(np.dot(X_train_11.T, X_train_11), x) - np.dot(X_train_11.T, y_train_11))
    grad_f_12 = lambda x : 2*(np.dot(np.dot(X_train_12.T, X_train_12), x) - np.dot(X_train_12.T, y_train_12))
    grad_f_13 = lambda x : 2*(np.dot(np.dot(X_train_13.T, X_train_13), x) - np.dot(X_train_13.T, y_train_13))
    grad_f_14 = lambda x : 2*(np.dot(np.dot(X_train_14.T, X_train_14), x) - np.dot(X_train_14.T, y_train_14))
    grad_f_15 = lambda x : 2*(np.dot(np.dot(X_train_15.T, X_train_15), x) - np.dot(X_train_15.T, y_train_15))
    grad_f_16 = lambda x : 2*(np.dot(np.dot(X_train_16.T, X_train_16), x) - np.dot(X_train_16.T, y_train_16))
    
    oracle1 = FirstOrderOracle(grad_f_1, dim_f)
    oracle2 = FirstOrderOracle(grad_f_2, dim_f)
    oracle3 = FirstOrderOracle(grad_f_3, dim_f)
    oracle4 = FirstOrderOracle(grad_f_4, dim_f)
    oracle5 = FirstOrderOracle(grad_f_5, dim_f)
    oracle6 = FirstOrderOracle(grad_f_6, dim_f)
    oracle7 = FirstOrderOracle(grad_f_7, dim_f)
    oracle8 = FirstOrderOracle(grad_f_8, dim_f)
    oracle9 = FirstOrderOracle(grad_f_9, dim_f)
    oracle10 = FirstOrderOracle(grad_f_10, dim_f)
    oracle11 = FirstOrderOracle(grad_f_11, dim_f)
    oracle12 = FirstOrderOracle(grad_f_12, dim_f)
    oracle13 = FirstOrderOracle(grad_f_13, dim_f)
    oracle14 = FirstOrderOracle(grad_f_14, dim_f)
    oracle15 = FirstOrderOracle(grad_f_15, dim_f)
    oracle16 = FirstOrderOracle(grad_f_16, dim_f)
    
    # oracles = [oracle1, oracle2]
    # oracles = [oracle1, oracle2, oracle3, oracle4]
    # oracles = [oracle1, oracle2, oracle3, oracle4, oracle5, oracle6, oracle7, oracle8]
    oracles = [oracle1, oracle2, oracle3, oracle4, oracle5, oracle6, oracle7, oracle8, oracle9, oracle10, oracle11, oracle12, oracle13, oracle14, oracle15, oracle16]
 
    # Testing for master-worker implementation
    x_init = np.zeros((dim_f,1))
    if rank == 0: start_time = time.time()
    grad = GradientDescent(oracles, max_iter=100000, alpha=lambda x:  0.0000000000008)   
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
    
    
def load_bikesharing():
    
    # source: http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip
    dataset = pd.read_csv('/Users/vtheophanous/parallel/datasets/Bike-Sharing-Dataset/hour.csv').values 
    # dataset = [list(d) for d in dataset]
    # print dataset.shape
    
    # get x training set
    X = np.asarray([d[2:-3].astype(float) for d in dataset]) 
    X = X[:, np.newaxis]
    X_train = X[:int(len(X)*0.8)] # train on 4/5 of the set
    # Now we need to add a column of ones to account for the intercept in finding a linear prediction
    X_train = np.matrix(X_train)
    # print X_train
    
    # get y training set
    # y = np.asarray([d[-3:].astype(float) for d in dataset]) # y is last 3 columns of dataset
    y = np.asarray([float(d[-1]) for d in dataset]) # last column
    y_train = y[:int(len(y)*0.8)] # train on 4/5 of the set
    # print y_train.shape
    y_train = np.matrix(y_train).T
    # print y_train
    
    return X_train, y_train


def load_twitter():
    # source: https://archive.ics.uci.edu/ml/machine-learning-databases/00248/regression.tar.gz
    dataset = np.loadtxt('/Users/vtheophanous/parallel/datasets/regression/Twitter/Twitter.data', delimiter=",")
    # print dataset.shape
    
    # get x training set
    X = np.asarray([d[:10] for d in dataset]) 
    X = X[:, np.newaxis]
    X_train = X[:int(len(X)*0.8)] # train on 4/5 of the set
    # X_train = X[:100000]
    # Now we need to add a column of ones to account for the intercept in finding a linear prediction
    X_train = np.matrix(X_train)
    # print X_train
    
    # get y training set
    y = np.asarray([float(d[-1]) for d in dataset]) # last column
    y_train = y[:int(len(y)*0.8)] # train on 4/5 of the set
    # y_train = y[:100000]
    # print y_train.shape
    y_train = np.matrix(y_train).T
    # print y_train
    
    return X_train, y_train

    
def load_random():
    X_train = np.random.rand(10000,1)
    X_train = np.matrix(X_train)
    # X_train = np.column_stack((X_train, np.ones(len(X_train))))
    
    y_train = np.random.rand(1,10000)
    y_train = np.matrix(y_train).T

    return X_train, y_train
    
if __name__ == "__main__":
    main()