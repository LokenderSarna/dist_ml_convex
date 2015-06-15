from dist_ml_convex import *
import numpy as np
import time
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# If the process is the master assign subsets of the dataset to each process via oracles
if rank == 0:
    
    # First we read from the csv via the pandas library
    data = pd.read_csv('../train.csv',skipinitialspace=1)
    Y_train = data['label']
    X_train = data.drop('label', 1)
    
    # We need to do some indexing here to assign evenly divided Xi's
    num_data_points = len(data)
    num_processes = size
    num_samples_per_process = num_data_points / size
    num_remainder_samples = num_data_points % size
    
    # Here we are subdividing the array to be broadcasted. It will be sent in the form: [Y, X1, X2, ... , Xn]
    data_bcast = []
    current_start_index = 0
    for index in range(1, size):
        if index != size:
            # Now lets split the data into even sections as a function of the size of our process world
            data_bcast.append((X_train._slice(slice(current_start_index,current_start_index+num_samples_per_process))).as_matrix())
            data_bcast.append((Y_train._slice(slice(current_start_index,current_start_index+num_samples_per_process))).as_matrix())
            current_start_index += num_samples_per_process
        else:
            data_bcast.append(X_train[:current_start_index])
            data_bcast.append(Y_train[:current_start_index])

# The other processes are receiving the data so they send None
else:
    data_bcast = None

# Here we are broadcasting the data subsets for the other processes to pick up. This avoids multiple csv reads.
data_bcast = comm.bcast(data_bcast, root=0)

# Now we need to generate the oracles
oracles = []

# First find the number of dimensions (columns) in each sample point
X1 = data_bcast[0]
dim_f = len(X1[0])

for process in range(0, size):
    # lambda x : X^Tx - Y, where x is the weights
    Xi_index = 2*process
    Yi_index = 2*process + 1
    grad_f = lambda x : np.subtract(np.multiply(np.transpose(data_bcast[Xi_index]), x), data_bcast[Yi_index])
    oracle = FirstOrderOracle(grad_f, dim_f)
    oracles.append(oracle)
    
if rank == 7:
    print len(data_bcast)
    print data_bcast[Xi_index].shape
    print data_bcast[Yi_index].shape
    xi = np.random.random((dim_f, 1))
    grad_f = oracles[7].grad_f
    print grad_f(xi)

# grad = GradientDescent(oracles)
# grad.execute()

