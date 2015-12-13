from dist_ml_convex import *
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import time

def main():
    
    # X_train, y_train = load_diabetes()
    # X_train, y_train = load_bikesharing()
    X_train, y_train = load_twitter()
    # X_train, y_train = load_random()
    # print X_train, y_train
    # print y_train
    # return 0
    
    dim_f = X_train.shape[1]
    # print dim_f

    
    # Initializing our gradient function for the first order oracle
    grad_f = lambda x : 2*(np.dot(np.dot(X_train.T, X_train), x) - np.dot(X_train.T, y_train))
    oracle = FirstOrderOracle(grad_f, dim_f)
    x_init = np.zeros((dim_f,1))
    start_time = time.time()
    grad = GradientDescentSingle(oracle, max_iter=100000, alpha=lambda x:  0.0000000000008)
    grad.execute()
    print("--Done: %s seconds --" % (time.time() - start_time))
    
    # # closed form solution
    # solution = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))
    # print "Closed form solution:"
    # print solution
    
def load_diabetes():
    # load dataset
    dataset = datasets.load_diabetes()

    # Split the data into training/testing sets
    X = dataset.data[:, np.newaxis]
    X_train = X[:-20]
    # Now we need to add a column of ones to account for the intercept in finding a linear prediction
    X_train = np.matrix(X_train)

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
    
    y_train = np.random.rand(1,10000)
    y_train = np.matrix(y_train).T

    return X_train, y_train
    
if __name__ == "__main__":
    main()