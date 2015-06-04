from mpi4py import MPI, MPE
import numpy as np
import math, sys, mpi4py.rc

# Defining a layer of abstraction for each fi. Which will likely end up being a cut where each fi is responsible for some subset of data a machine learning algorithm needs to sift through
class FirstOrderOracle:

    # Where f is the function and grad_f is its gradient, f_dim
    def __init__(self, f, grad_f, dim_f):
        self.f = f
        self.grad_f = grad_f
        self.dim_f = dim_f

# Defaults for gradient descent parameters
EPSILON_DEFAULT = 0.001
MAX_ITER_DEFAULT = 100000
ALPHA_DEFAULT = 0.001

# Gradient_descent will take a convex problem as input and find the optimal x* 
class GradientDescent:
    
    # Parameters taken as input; alpha: the step size function, epsilon the stopping critera, epsilon: the stop criterion, max_iter: upper limit on the number of iterations
    def __init__(self, oracles, alpha=None, epsilon=None, max_iter=None, x_init=None):
        
        # Here our gradient descent initialization will need to be assigned as the master
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Needed variables for the MPI processing
        self.comm = comm
        self.rank = rank
        self.size = size
        
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
            if self.rank == 0:
                data_bcast = np.random.random((self.dim_f, 1))
            else:
                data_bcast = None
            # There might be a higher communication cost of using a randomly initialized x vector, but for now lets leave it
            data_bcast = self.comm.bcast(data_bcast, root=0)     
            self.x = data_bcast
            
        else:
            self.x = x_init
        
        if self.size != len(self.oracles) + 1 and self.rank == 0:
            # Note later you should add a real error to be triggered
            print "ERROR: Your script call should be in the form:"
            print "mpirun -np <int: number of oracles + 1> python <string: name of script>"
                
        
        # Now if the process call being made here is of rank above 0 we know it is a child and needs to be assigned the correct oracle with associated grad_f
        if self.rank > 0:
            # Note that we are assigned the oracles at the -1 index to account for the master at index 0
            self.oracle = oracles[rank - 1]
            
    
    # Now we need our method to start the process calls
    def execute(self):
        
        num_iter = 0
        cont_iter = True
        sol_found = False
        
        # First we will seperate the worker and master executions
        if self.rank == 0:
            
            # Here we initiate the loop to find an x solution to the gradient descent
            while cont_iter == True:
        
                num_iter += 1
                
                # Now we need to either send a self.x vector or flag that child processes should finish their executions
                if num_iter > self.max_iter or sol_found == True:
                    # If we can't find a value within the max iteration limit stop the execution
                    if num_iter > self.max_iter:
                        print "Stopping after reaching iteration limit."
                    else:
                        print "Gradient descent has found a solution:"
                        print self.x
                    cont_iter = False
                    data_bcast = cont_iter
                else:
                    # Send the current x vector to the children
                    data_bcast = self.x
        
                # Initialize our masters summation of the gradients
                g_x = np.zeros_like(self.x)
        
                # Now broadcast our self.x value so the other worker processes can compute their oracles gradient
                data_bcast = self.comm.bcast(data_bcast, root=0)
                
                # Here we need to check that data_bcast just sent wasn't a flag to stop the worker execution. If it was then we need to stop our master execution
                if cont_iter == False:
                    if sol_found == True:
                        return self.x
                    else:
                        return None
        
                x_out = np.zeros_like(self.x)
                # Perform the reduce to find the g_x sum of gradient vector        
                self.comm.Reduce([x_out, MPI.DOUBLE], [g_x, MPI.DOUBLE], op=MPI.SUM, root=0)    
        
                # Here we do the update if we aren't within the epsilon value
                if np.linalg.norm(g_x) > self.epsilon:
                    # Note that fi is a function that goes from fi: R^d -> R^d, xk+1 = xk - alpha(num_iter)*f(xk)
                    self.x = np.subtract(self.x, np.multiply(self.alpha(num_iter), g_x))
                else:
                    sol_found = True
        
        # Worker execution is as follows
        else:
            
            # Here we initiate the loop to find an x solution to the gradient descent
            while cont_iter == True:
        
                # For our worker execution case these variables do not need to send values and are thus assigned to None
                data_bcast = None
                g_x = None
        
                # Now broadcast our self.x value so the other worker processes can compute their oracles gradient
                data_bcast = self.comm.bcast(data_bcast, root=0)
        
                # Check to make sure we haven't been flagged to stop worker execution
                if isinstance(data_bcast, bool):
                    cont_iter = False
                    continue
                
                x_out = np.zeros_like(self.x)
                # Now if the rank is above 0 the process is the child and should compute the gradient at current x
                x_out = self.oracle.grad_f(data_bcast)
            
                # Perform the reduce to find the g_x sum of gradient vector        
                self.comm.Reduce([x_out, MPI.DOUBLE], [g_x, MPI.DOUBLE], op=MPI.SUM, root=0)