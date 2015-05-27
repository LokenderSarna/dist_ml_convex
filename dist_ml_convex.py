import numpy as np
import math

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
EPSILON_DEFAULT = 0.0001
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
                # Note that fi is a function that goes from fi: R^d -> R^d, xk+1 = xk = alpha(num_iter)*f(xk)
                self.x = np.subtract(self.x, np.multiply(self.alpha(num_iter),g_x))
            else:
                cont_iter = False
                print "Gradient descent has found a solution:"
                print self.x
                return self.x
            
            # If we can't find a value within the max iteration limit return None
            if num_iter > self.max_iter:
                print "Stopping after reaching iteration limit."
                cont_iter = False
                #NOTE: add a error message for return or some flag