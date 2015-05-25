import numpy as np
import math

# Here we define the class convex_problem to build a convex optimization problem to be sent to gradient descent
class convex_problem:
	
    # Take input basic parameters; d: x num vector dimensions, n: num terms in sum, lambdas: functions fis, x0: initial x value 
    def __init__(self, d, n, lambdas, x0=None):
        self.d = d
        self.n = n
        self.lambdas = lambdas
        if x0 is None:
            self.x0 = np.zeros(d)
        else:
            self.x0 = x0

# Defaults for gradient descent parameters
EPSILON_DEFAULT = 0.01
MAX_ITER_DEFAULT = 100000

# Since alpha will be a function our default will also be a function to match type checking
def ALPHA_DEFAULT(x):
     return 0.5

# Gradient_descent will take a convex problem as input and find the optimal x* 
class gradient_descent:
    
    # Parameters taken as input; alpha: the step size function, epsilon the stopping critera, epsilon: the stop criterion, max_iter: upper limit on the number of iterations
    def __init__(self, convex_problem, alpha=None, epsilon=None, max_iter=None):
        self.convex_problem = convex_problem
        self.d = convex_problem.d
        self.n = convex_problem.n
        self.lambdas = convex_problem.lambdas
        self.x = convex_problem.x0
        if alpha is None:
            self.alpha = ALPHA_DEFAULT(0)
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
    
    # Here we will actually execute the algorithm
    def execute(self):
        
        # While the l2 norm of the gradient of the vector x is less then epsilon (the stopping criterion), keep going
        
    
    # A method to find the p=2 norm of a set of lambda function
    # NOTE: can we have the lambda take input an array?????
    def l2norm(self, x, lambdas):
        sum = 0
        for fi in lambdas:
            sum += math.pow(fi(x), 2)
        return math.sqrt(sum)
            
        
        
        