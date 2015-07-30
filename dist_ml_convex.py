from mpi4py import MPI
import numpy as np
import math, sys, time, csv

# Defining a layer of abstraction for each fi. Which will likely end up being a cut where each fi is responsible for some subset of data a machine learning algorithm needs to sift through
class FirstOrderOracle:

    # Where f is the function and grad_f is its gradient, f_dim
    def __init__(self, grad_f, dim_f, f=None):
        if f is None:
            f = None
        else:
            self.f = f
        self.grad_f = grad_f
        self.dim_f = dim_f

# Defaults for gradient descent parameters
EPSILON_DEFAULT = 0.001
MAX_ITER_DEFAULT = 10000
ALPHA_DEFAULT = 0.001

# Gradient_descent will take a convex problem as input and find the optimal x* 
class GradientDescent:
    
    # Parameters taken as input; alpha: the step size function, epsilon the stopping critera, epsilon: the stop criterion, max_iter: upper limit on the number of iterations
    def __init__(self, oracles, alpha=None, epsilon=None, max_iter=None, x_init=None, lipschitz=None):
        
        # Here our gradient descent initialization will need to be assigned as the master
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.oracles = oracles
        self.dim_f = self.oracles[0].dim_f
        
        # Define whether or not to use the lipschitz values for decreasing alpha
        if lipschitz is True:
            self.lipschitz = True
        else:
            self.lipschitz = False
        
        # Since alpha will be a function our default will also be a function to match type checking
        if alpha is None:
            self.alpha = (lambda x : ALPHA_DEFAULT)
        else:
            self.alpha = alpha
        
        # Assigning epsilon value which is the size of our solution space
        if epsilon is None:
            self.epsilon = EPSILON_DEFAULT
        else:
            self.epsilon = epsilon
        
        # The maximum number of iterations before the algorithm terminate without a solution in the epsilon space
        if max_iter is None:
            self.max_iter = MAX_ITER_DEFAULT
        else:
            self.max_iter = max_iter
        
        # We will randomly choose our x initial values for gradient descent, the x values being the values we are optimizing
        if x_init is None:
            if self.rank == 0:
                data_bcast = np.random.random((self.dim_f, 1))
            else:
                data_bcast = None
            # There might be a higher communication cost of using a randomly initialized x vector, but for now lets leave it
            data_bcast = self.comm.bcast(data_bcast, root=0)     
            self.x = data_bcast
            self.x_previous = np.zeros((self.dim_f, 1))
        else:
            self.x = x_init
            self.x_previous = np.zeros((self.dim_f, 1))
            
        # Just to initialize the values I'm going to assign the current g_x and the previous g_x as the current x
        self.g_x = self.x
        self.g_x_previous = self.x
        self.L = 0
        
        # Now if lipschitz has been switched on we need to define the alpha to be a function of L
        if self.lipschitz == True:
            self.alpha = (lambda iteration : 1 / (self.L * np.sqrt(iteration)))
                
        # Check statement to see if the user has inputed the correct number of processes
        if self.size != len(self.oracles) + 1 and self.rank == 0:
            # Note later you should add a real error to be triggered
            print "ERROR: Your script call should be in the form:"
            print "mpirun -np <int: number of oracles + 1> python <string: name of script>"
        
        # Now if the process call being made here is of rank above 0 we know it is a child and needs to be assigned the correct oracle with associated grad_f
        if self.rank > 0:
            # Note that we are assigned the oracles at the -1 index to account for the master at index 0
            self.oracle = oracles[self.rank - 1]
        
    
    # Now we need our method to start the process calls
    def execute(self):
        
        num_iter = 0
        cont_iter = True
        sol_found = False
        
        bcast_time_master = 0
        reduce_time_master = 0
        
        bcast_time_worker = 0
        reduce_time_worker = 0
        
        
        # First we will seperate the worker and master executions
        if self.rank == 0:
            
            # Here we initiate the loop to find an x solution to the gradient descent
            while cont_iter == True:
                num_iter += 1
                
                # Now we need to either send a self.x vector or flag that child processes should finish their executions
                if num_iter > self.max_iter or sol_found == True:
                    # If we can't find a value within the max iteration limit stop the execution
                    if num_iter > self.max_iter:
                        print "Stopping after reaching iteration limit. Here was the final weights found:"
                        print self.x
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
                
                if self.lipschitz == True:
                    # Now update the lip value if needed
                    self.g_x_previous = self.g_x
                    self.g_x = g_x
                    
                    # Now if we are on the first iteration we need to just use the current values and no previous ones
                    if num_iter == 1:
                        self.L = self.g_x / self.x
                    # Otherwise use the difference between the gradients to find the lipschitz constant
                    else:
                        self.L = np.linalg.norm(self.g_x - self.g_x_previous) / np.linalg.norm(self.x - self.x_previous)
                    # Now we do book keeping on the previous x
                    self.x_previous = self.x
                    
                    
                # Here we do the update if we aren't within the epsilon value
                if np.linalg.norm(g_x) > self.epsilon:                        
                    # Note that fi is a function that goes from fi: R^d -> R^d, xk+1 = xk - alpha(num_iter)*f(xk)
                    self.x = np.subtract(self.x, np.multiply(self.alpha(num_iter), g_x))
                    
                    if num_iter%1000 == 0:
                        print "The current self.x value is:"
                        print self.x
                        print "np.linalg.norm(g_x) = %f" %np.linalg.norm(g_x)
                    
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