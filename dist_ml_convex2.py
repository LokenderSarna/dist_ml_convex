from mpi4py import MPI
import numpy as np
import math, sys

# Main method for executing our iteration
def main():
    
    # Define main execute
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Now we decide wether this is a master or worker call of this script
    if rank == 0:
        master()
        data = [(i+1)**2 for i in range(size)]
        
    else:
        worker()
        data = None
    data = comm.scatter(data, root=0)
    assert data == (rank+1)**2
    print rank
    print data

    
# Define master execute which will be gradient descent
def master():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return None


# Define worker execute which will compute the gradient descent at a given point
def worker():
    
    # We get the gradient associated with the oracle at index = rank
    # ???? how do I make this call to get the oracles???? do I assume its already an instance and just call oracles ????
    #grad_f = GradientDescent.self.oracles[rank].grad_f
        
    #comm = MPI.COMM_WORLD
    #rank = comm.Get_Rank()

    #if rank == 2:
       #print "I'm rank 2 and I like bacon!"
    
    return None

    
# Defining a layer of abstraction for each fi. Which will likely end up being a cut where each fi is responsible for some subset of data a machine learning algorithm needs to sift through
class FirstOrderOracle:

    # Where f is the function and grad_f is its gradient
    def __init__(self, f, grad_f, f_dim, x_init=None):
        self.f = f
        self.grad_f = grad_f
        self.f_dim = f_dim
        if x_init == None:
            self.x_init = np.random.random((self.f_dim, 1))
        else:
            self.x_init = x_init
     
# Defaults for gradient descent parameters
EPSILON_DEFAULT = 0.001
MAX_ITER_DEFAULT = 10000000
ALPHA_DEFAULT = 0.001

# GradientDescent will take a convex problem as input and find the optimal x* 
class GradientDescent:
    
    # Parameters taken as input; oracles: the functions at each worker node, alpha: the step size function, epsilon the stopping critera, epsilon: the stop criterion, max_iter: upper limit on the number of iterations
    def __init__(self, oracles, alpha=None, epsilon=None, max_iter=None):
        
        self.oracles = oracles
        self.f_dim = self.oracles[0].f_dim
        self.x = self.oracles[0].x_init
        
        # Since alpha will be a function our default will also be a function to match type checking
        if alpha is None:
            self.alpha = (lambda x : ALPHA_DEFAULT)
        else:
            self.alpha = alpha
        if epsilon is None:
            self.epsilon = EPSILON_DEFAULT
        else:
            self.epsilon = epsilon
        if max_iter is None:
            self.max_iter = MAX_ITER_DEFAULT
        else:
            self.max_iter = max_iter         
        
    def execute(self):
        print "At execute In Gradient Descent!"
            

if __name__=="__main__":
    main()