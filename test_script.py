from dist_ml_convex import *
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dim_f = 10

# First lets branch for depending on if this call is the master or worker
if rank == 0:
    
    print "The number of processes running: comm.Get_size() = %d" %comm.Get_size()
    
    Q1 = np.random.random((dim_f,dim_f))
    Q1 = 0.5*(Q1 + np.transpose(Q1))
    Q1 = np.linalg.matrix_power(Q1, 2)
    Q2 = np.random.random((dim_f,dim_f))
    Q2 = 0.5*(Q2 + np.transpose(Q2))
    Q2 = np.linalg.matrix_power(Q2, 2)
    Q3 = np.random.random((dim_f,dim_f))
    Q3 = 0.5*(Q3 + np.transpose(Q3))
    Q3 = np.linalg.matrix_power(Q3, 2)
    
    q1 = np.random.random((dim_f,1))
    q2 = np.random.random((dim_f,1))
    q3 = np.random.random((dim_f,1))
        
    Q = Q1 + Q2 + Q3
    q = q1 + q2 + q3

    print "Closed form Solution: "
    print - np.dot(np.linalg.inv(Q), q)

    data_bcast = [Q1, Q2, Q3, q1, q2, q3]

else:
    data_bcast = None

# Here we needed only one generation of random matrices. If we made instances in each call, the matrices would be different in each.
data_bcast = comm.bcast(data_bcast, root=0)

Q1 = data_bcast[0]
Q2 = data_bcast[1]
Q3 = data_bcast[2]
q1 = data_bcast[3]
q2 = data_bcast[4]
q3 = data_bcast[5]

f1 = lambda x : (1/2)*(np.dot(np.dot(np.transpose(x), Q1), x)) + np.dot(q1, x)
f2 = lambda x : (1/2)*(np.dot(np.dot(np.transpose(x), Q2), x)) + np.dot(q2, x)
f3 = lambda x : (1/2)*(np.dot(np.dot(np.transpose(x), Q3), x)) + np.dot(q3, x)

grad_f1 = lambda x : np.dot(Q1, x) + q1
grad_f2 = lambda x : np.dot(Q2, x) + q2
grad_f3 = lambda x : np.dot(Q3, x) + q3

oracle1 = FirstOrderOracle(grad_f1, dim_f, f=f1) 
oracle2 = FirstOrderOracle(grad_f2, dim_f, f=f2) 
oracle3 = FirstOrderOracle(grad_f3, dim_f, f=f3) 


t0 = time.time()
oracles = [oracle1, oracle2, oracle3]
grad = GradientDescent(oracles)
t1 = time.time()
print "rank = %d initialization time = %f" %(rank, (t1 - t0))

t0 = time.time()
grad.execute()
t1 = time.time()
print "rank = %d 'grad.execute()' execution time = %f" %(rank, (t1 - t0))