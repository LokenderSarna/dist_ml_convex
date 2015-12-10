from dist_ml_convex import *
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import time

def main():
    
    dim_f = 11
    
    X_train, y_train = load_dataset(dim_f)
    
    # Initializing our gradient function for the first order oracle
    grad_f = lambda x : 2*(np.dot(np.dot(X_train.T, X_train), x) - np.dot(X_train.T, y_train))
    
    oracle = FirstOrderOracle(grad_f, dim_f)
    
    # Testing for single process implementation
    x_init = np.zeros((dim_f,1))
    start_time = time.time()
    grad = GradientDescentSingle(oracle, max_iter=1000000, x_init=x_init, alpha=lambda x: 0.001)   
    grad.execute()
    print("--Done: %s seconds --" % (time.time() - start_time))
    
def load_dataset(dim):
    # load dataset
    dataset = datasets.load_diabetes()
    # dataset = load_bikesharing()

    # use dim-1 number of features
    X = dataset.data[:, np.newaxis]
    X_temp = X[:, :, :(dim-1)]
    # print X_temp.shape
    # X_temp = []
    
    # Split the data into training/testing sets
    # X_train = X_temp[:int(len(X_temp)*0.8)] # train on 4/5 of the set
    X_train = X_temp[:-20]
    # Now we need to add a column of ones to account for the intercept in finding a linear prediction
    X_train = np.matrix(X_train)
    X_train = np.column_stack((X_train, np.ones(len(X_train))))
    # print X_train.shape

    # Split the targets into training/testing sets
    y_train = dataset.target[:-20]
    y_train = np.matrix(y_train).T
    #y_test = dataset.target[-20:]
    # print y_train.shape
    
    
    return X_train, y_train
    
    
def load_bikesharing():
    
    data = pd.read_csv('filepath').values
    print data
    # return data
    
    
if __name__ == "__main__":
    main()