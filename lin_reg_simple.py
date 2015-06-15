from dist_ml_convex import *
import numpy as np
import time
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Method to flatten a matrix down for broadcasting with Bcast returns a simple flat array
def flatten(A):
    return np.array(A.flatten().tolist(), dtype=np.int16)

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
    
    # First we read from the csv via the pandas library
    data = pd.read_csv('../train.csv', skipinitialspace=1)
    Y_train = data['label']
    X_train = data.drop('label', 1)
    
    # Now split each data matrix needed into a flattened array for broadcasting
    X1 = flatten(X_train._slice(slice(0,10000)).as_matrix())
    Y1 = flatten(Y_train._slice(slice(0,10000)).as_matrix())
    X2 = flatten(X_train._slice(slice(10000,20000)).as_matrix())
    Y2 = flatten(Y_train._slice(slice(10000,20000)).as_matrix())
    X3 = flatten(X_train._slice(slice(20000,30000)).as_matrix())
    Y3 = flatten(Y_train._slice(slice(20000,30000)).as_matrix())
    X4 = flatten(X_train._slice(slice(30000,42000)).as_matrix())
    Y4 = flatten(Y_train._slice(slice(30000,42000)).as_matrix())            

# The other processes are receiving the data so they send None
else:
    X1 = np.empty(784*10000, dtype=np.int16)
    Y1 = np.empty(1*10000, dtype=np.int16)
    X2 = np.empty(784*10000, dtype=np.int16)
    Y2 = np.empty(1*10000, dtype=np.int16)
    X3 = np.empty(784*10000, dtype=np.int16)
    Y3 = np.empty(1*10000, dtype=np.int16)
    X4 = np.empty(784*12000, dtype=np.int16)
    Y4 = np.empty(1*12000, dtype=np.int16)

# Here we are broadcasting the data subsets for the other processes to pick up. This avoids multiple csv reads.
comm.Bcast([X1, MPI.DOUBLE], root=0)
comm.Bcast([Y1, MPI.DOUBLE], root=0)
comm.Bcast([X2, MPI.DOUBLE], root=0)
comm.Bcast([Y2, MPI.DOUBLE], root=0)
comm.Bcast([X3, MPI.DOUBLE], root=0)
comm.Bcast([Y3, MPI.DOUBLE], root=0)
comm.Bcast([X4, MPI.DOUBLE], root=0)
comm.Bcast([Y4, MPI.DOUBLE], root=0)

# Declare the dimensions of f as 784 which is the number of columns at each sample point
dim_f = 784

# Now each subprocesses is delegated to its own operations set. Where rank == 0 is the master and all others are workers
if rank == 0:
    print "Master here!"

# Worker processes follow this branch of execution
else:

        
    if rank == 1:
    
        X = unflatten(X1, 10000, 784)
        Y = unflatten(Y1, 10000, 1)
    
        print X.shape
        
        XT = X.T
        print XT.shape

        XTX = np.dot(XT, X)
        
        print XTX.shape
        
        print Y.shape
        # XtX = X.T.dot(X)
        # print XtX.shape
        # XtXY = np.dot(XtX, Y)
        # print XtXY.shape

# Now we need to generate the grad_f lambdas

# if rank == 0:
#
#     print getsizeof(X1)
#     print 1
#     print X1.shape
#     print np.transpose(X1).shape
#     print 2
#     X1tX1 = X1.T.dot(X1)
#     #X1tX1 = np.dot(np.transpose(X1), X1)
#     print 3
#     print X1tX1.shape
#     print 4
#     wi = np.random.random((dim_f, 1))
#     X1tX1w = np.dot(X1tX1, wi)
#     print 5
#     print X1tX1w.shape
#
    # grad_f1 = lambda x : 2*(np.dot(np.dot(np.transpose(X1), X1), x) - np.dot(np.transpose(X1), Y1))
    # print grad_f1(wi)

# grad = GradientDescent(oracles)
# grad.execute()

