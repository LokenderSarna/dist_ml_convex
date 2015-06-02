from mpi4py import MPI
import numpy as np
import math, sys

# Defining a layer of abstraction for each fi. Which will likely end up being a cut where each fi is responsible for some subset of data a machine learning algorithm needs to sift through
class FirstOrderOracle:

    # Where f is the function and grad_f is its gradient, f_dim
    def __init__(self, f, grad_f, dim_f):
        self.f = f
        self.grad_f = grad_f
        self.dim_f = dim_f

# Defaults for gradient descent parameters
EPSILON_DEFAULT = 0.001
MAX_ITER_DEFAULT = 10000000
ALPHA_DEFAULT = 0.001

# Gradient_descent will take a convex problem as input and find the optimal x* 
class GradientDescent:
    
    # Parameters taken as input; alpha: the step size function, epsilon the stopping critera, epsilon: the stop criterion, max_iter: upper limit on the number of iterations
    def __init__(self, oracles, alpha=None, epsilon=None, max_iter=None, x_init=None):
        self.oracles = oracles
        self.dim_f = self.oracles[0].dim_f
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
        if x_init == None:
            # We will randomly choose our x initial values for gradient descent
            self.x = np.random.random((self.dim_f, 1))
        else:
            self.x = x_init
        
        # Here our gradient descent initialization will need to be assigned as the master
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Needed variables for the MPI processing
        self.comm = comm
        self.rank = rank
        self.size = size
        
        if self.size != len(self.oracles) + 1:
            # Note later you should add a real error to be triggered
            print "ERROR: Your script call should be in the form:"
            print "mpirun -np <int: number of oracles + 1> python <string: name of script>"
        
        print "The size is %d" %self.size
        
        
        # Now if the process call being made here is of rank above 0 we know it is a child and needs to be assigned the correct oracle with associated grad_f
        if self.rank > 0:
            # Note that we are assigned the oracles at the -1 index to account for the master at index 0
            self.oracle = oracles[rank - 1]
            
    
    # Now we need our method to start the process calls
    def execute(self):
        
        # First lets branch for depending on if this call is the master or worker
        if self.rank == 0:
            # If this is the master we need to send the current x value to the children
            data_bcast = self.x
            print "I am the master and I hold all the x value. I'll broadcast it soon."
            # Initialize our masters summation of the gradients
            g_x = np.zeros_like(self.x)
            
        else:
            data_bcast = None
            g_x = None
        
        
        num_iter = 1
        
        # Now broadcast our self.x value so the other worker processes can compute their oracles gradient
        data_bcast = self.comm.bcast(data_bcast, root=0)
        
        x_out = np.zeros_like(self.x)
        # Now if the rank is above 0 the process is the child and should compute the gradient at current x
        if self.rank > 0:
            x_out = self.oracle.grad_f(data_bcast)
            print "I have rank = %d and have computed an output for grad_f(x): " %self.rank
        
        self.comm.Reduce([x_out, MPI.DOUBLE], [g_x, MPI.DOUBLE], op=MPI.SUM, root=0)
        
        print "g_x at process = %d:" %self.rank
        print g_x        
        
        if self.rank == 0:
        
            # Here we do the update if we aren't within the epsilon value
            if np.linalg.norm(g_x) > self.epsilon:
                # Note that fi is a function that goes from fi: R^d -> R^d, xk+1 = xk - alpha(num_iter)*f(xk)
                self.x = np.subtract(self.x, np.multiply(self.alpha(num_iter), g_x))
                print "updated self.x"
            else:
                cont_iter = False
                print "Gradient descent has found a solution:"
                print self.x
            
                return self.x
        
        #print "g_x is:"
        #print g_x
        
        
        
        
        
        
        
        
        
        
        