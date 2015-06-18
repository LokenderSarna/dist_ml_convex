from dist_ml_convex import *
import numpy as np
import time
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Method to flatten a matrix down for broadcasting with Bcast returns a simple flat array
def flatten(A):
    return np.array(A.flatten().tolist(), dtype=np.float64)

# Going to define a method to unflatten an input list/array A and output a 2d array of dimensions n x m
def unflatten(A, n, m):
    num_elements = len(A)
    # Check to make sure the number of elements match
    if num_elements != n*m:
        return None
    # B the array we will eventually return after filling
    B = np.empty([n,m], dtype=int)
    
    # Iterate and fill
    row_index = 0
    for index in range(0, num_elements, m):
        B[row_index] = A[index:(index+m)]
        row_index += 1
    
    # Return the the unflattened array  
    return B 

# If the process is the master assign subsets of the dataset to each process via oracles
if rank == 0:
    
    # Gather data from train and test csv 
    data_train = pd.read_csv('../train.csv', skipinitialspace=1, index_col=0, parse_dates=True)

    # Explode datetime into hour, dayofweek, and month
    data_train['hour'] = data_train.index.hour
    data_train['dayofweek'] = data_train.index.weekday
    data_train['month'] = data_train.index.month

    # Build a dataset with its matching y outputs
    X_selected_cols = [u'weather',u'dayofweek',u'hour',u'season',u'holiday',u'workingday',u'temp',u'atemp',u'humidity',u'windspeed']
    X_train = data_train[X_selected_cols]
    Y_train = data_train['count']
    
    # Now split each data matrix needed into a flattened array for broadcasting
    X1 = flatten(X_train._slice(slice(0,2500)).as_matrix())
    Y1 = flatten(Y_train._slice(slice(0,2500)).as_matrix())
    X2 = flatten(X_train._slice(slice(2500,5000)).as_matrix())
    Y2 = flatten(Y_train._slice(slice(2500,5000)).as_matrix())
    X3 = flatten(X_train._slice(slice(5000,7500)).as_matrix())
    Y3 = flatten(Y_train._slice(slice(5000,7500)).as_matrix())
    X4 = flatten(X_train._slice(slice(7500,10000)).as_matrix())
    Y4 = flatten(Y_train._slice(slice(7500,10000)).as_matrix())      

# The other processes are receiving the data so they send None
else:
    X1 = np.empty(25000, dtype=np.float64)
    Y1 = np.empty(2500, dtype=np.float64)
    X2 = np.empty(25000, dtype=np.float64)
    Y2 = np.empty(2500, dtype=np.float64)
    X3 = np.empty(25000, dtype=np.float64)
    Y3 = np.empty(2500, dtype=np.float64)
    X4 = np.empty(25000, dtype=np.float64)
    Y4 = np.empty(2500, dtype=np.float64)    

# Here we are broadcasting the data subsets for the other processes to pick up. This avoids multiple csv reads.
comm.Bcast([X1, MPI.DOUBLE], root=0)
comm.Bcast([Y1, MPI.DOUBLE], root=0)
comm.Bcast([X2, MPI.DOUBLE], root=0)
comm.Bcast([Y2, MPI.DOUBLE], root=0)
comm.Bcast([X3, MPI.DOUBLE], root=0)
comm.Bcast([Y3, MPI.DOUBLE], root=0)
comm.Bcast([X4, MPI.DOUBLE], root=0)
comm.Bcast([Y4, MPI.DOUBLE], root=0)

# Now deal with the correct sizing of the arrays
subset_len = 2500        

# Now unflatten the broadcasted data
X1 = unflatten(X1, subset_len, 10)
Y1 = unflatten(Y1, subset_len, 1)
X2 = unflatten(X2, subset_len, 10)
Y2 = unflatten(Y2, subset_len, 1)
X3 = unflatten(X3, subset_len, 10)
Y3 = unflatten(Y3, subset_len, 1)
X4 = unflatten(X4, subset_len, 10)
Y4 = unflatten(Y4, subset_len, 1)

# Define gradients for oracles
X = X1
XT = X1.T
Y = Y1
grad_f1 = lambda x : 2*np.dot(np.dot(XT, X), x) - np.dot(XT, Y)
X = X2
XT = X2.T
Y = Y2
grad_f2 = lambda x : 2*np.dot(np.dot(XT, X), x) - np.dot(XT, Y)
X = X3
XT = X3.T
Y = Y3
grad_f3 = lambda x : 2*np.dot(np.dot(XT, X), x) - np.dot(XT, Y)
X = X4
XT = X4.T
Y = Y4
grad_f4 = lambda x : 2*np.dot(np.dot(XT, X), x) - np.dot(XT, Y)

w = np.random.random((10, 1))

# Declare the dimensions of f, which is the number of columns at each sample point
dim_f = 10

oracle1 = FirstOrderOracle(grad_f1, dim_f)
oracle2 = FirstOrderOracle(grad_f2, dim_f)
oracle3 = FirstOrderOracle(grad_f3, dim_f)
oracle4 = FirstOrderOracle(grad_f4, dim_f)

oracles = [oracle1, oracle2, oracle3, oracle4]
grad = GradientDescent(oracles, alpha=(lambda x : 0.0000000168), max_iter=1000000, epsilon=10000)
grad.execute()
