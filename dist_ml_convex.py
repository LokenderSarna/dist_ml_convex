from mpi4py import MPI
import numpy as np
import math, sys, inspect, itertools

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

# Gradient_descent will take a convex problem as input and find the optimal x* 
class GradientDescent:
    
    # Parameters taken as input; alpha: the step size function, epsilon the stopping critera, epsilon: the stop criterion, max_iter: upper limit on the number of iterations
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
    
    # define a process that will take our derivative lambda function and continually compute values when needed
    def spawn(self):
        
        print self.oracles
        print self.x
        print [self.x, self.oracles]
        
        """
        #get the text value for grad_f to send as an argument for the child process
        self.oracles[0].grad_f
        grad_f_string = inspect.getsource(self.oracles[0].grad_f)
        grad_f = self.oracles[0].grad_f
        print grad_f
        print "\n"
        print "grad_f_string:"
        print grad_f_string
        
        var = 40
        
        ARGS = ['-c', '\n'.join([
                'import sys',
                'from mpi4py import MPI',
                'parent = MPI.Comm.Get_parent()',
                '#rank = MPI.Comm.Get_rank()',
                '#%s' % grad_f,
                '#print var',
                'print "We made it to a child process!!!"',
                '#print grad_f1',
                'parent.Disconnect()',
                ])]
        """
        
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['child.py'], maxprocs=len(self.oracles))
        buf = [self.x, self.oracles]
        comm.Bcast(buf, root=MPI.ROOT)
        
        gradient = numpy.array(0.0, 'd')
        comm.Reduce(None, [new_x, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
        self.x = new_x
        
        comm.Disconnect()
        
        return None
    
    # Master process that controls the worker processes execution of finding each gradient            
    def master(self):
        
        self.spawn()
        
        return None
        # Create the communication instance
        #self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['worker.py'])
            
        # declare the buffer holding the current value x(k)
        #current_buffer = numpy.array(current_x_of_k, 'd')
        
        # disconnect from the communicator
        #comm.Disconnect()    
        
        # Returns the solution x
            
    # Here we will actually execute the algorithm and return the solution if one is reached
    def execute(self):
                      
        # Iterate through until the x vector values allow the summed up fis to fall within epsilon
        cont_iter = True
        num_iter = 0
        
        while cont_iter == True:
            
            # Increment the number of iterations, also note alpha is a function of num_iter
            num_iter += 1
            
            # Where g_x is the summation of all of our fi's at each x
            g_x = np.ndarray(shape=(self.f_dim,1))
            g_x.fill(0)
            for oracle in self.oracles:
                g_x = np.add(g_x, oracle.grad_f(self.x))
            
            # Here we do the update if we aren't within the epsilon value
            if np.linalg.norm(g_x) > self.epsilon:
                # Note that fi is a function that goes from fi: R^d -> R^d, xk+1 = xk - alpha(num_iter)*f(xk)
                self.x = np.subtract(self.x, np.multiply(self.alpha(num_iter),g_x))
            else:
                cont_iter = False
                print "Gradient descent has found a solution:"
                print self.x
                
                self.master()
                
                return self.x
            
            # If we can't find a value within the max iteration limit return None
            if num_iter > self.max_iter:
                print "Stopping after reaching iteration limit."
                cont_iter = False
                # NOTE: add a error message for return or some flag

"""
# Class to define the children processes of masters             
class Child:
    
    # The init method will prep the child for spawning
    def __init__(self, oracle):
        self.oracle = oracle
        
    # Method to execute the computation needed for reducing 
    def spawn(self):
        ARGS = ['-c', '\n'.join([
                'import sys',
                'from mpi4py import MPI',
                'from dist_ml_convex import *'
                'child%s.execute()', % child_id
                'parent.Disconnect()',
                ])]
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=ARGS)
        
        
    # Method to disconnect
    def disconnenct(self):
        parent.Disconnect()
"""    

"""
# Master process that controls the worker processes execution of finding each gradient            
def Master:
    
    # We should only need the oracles to control the processes and return results
    def __init__(self, oracles):
        self.oracles = oracles
        
        # create the communication instance
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['worker.py'])
        
    # Here we will find the summation of gradients at x using sub processes
    def find_grad(x):
        return None
"""