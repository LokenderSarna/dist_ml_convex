import numpy as np
import math

# Here we define the class convex_problem to build a convex optimization problem to be sent to gradient descent
class ConvexProblem:
	
    # Take input basic parameters; d: x num vector dimensions, n: num terms in sum, lambdas: functions fis, x0: initial x value 
    def __init__(self, d, lambdas, x0=None):
        self.d = d
        self.n = len(lambdas)
        self.lambdas = lambdas
        if x0 is None:
            self.x0 = np.random.random((self.d,1))
        else:
            self.x0 = x0
                    
# Defaults for gradient descent parameters
EPSILON_DEFAULT = 0.001
MAX_ITER_DEFAULT = 10000000
ALPHA_DEFAULT = 0.001

# Gradient_descent will take a convex problem as input and find the optimal x* 
class GradientDescent:
    
    # Parameters taken as input; alpha: the step size function, epsilon the stopping critera, epsilon: the stop criterion, max_iter: upper limit on the number of iterations
    def __init__(self, convex_problem, alpha=None, epsilon=None, max_iter=None):
        self.convex_problem = convex_problem
        self.d = convex_problem.d
        self.n = convex_problem.n
        self.lambdas = convex_problem.lambdas
        self.x = convex_problem.x0
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
                
    # Here we will actually execute the algorithm
    def execute(self):
                      
        # Iterate through until the x vector values allow the summed up fis to fall within epsilon
        cont_iter = True
        num_iter = 0
        
        while cont_iter == True:
            
            # Increment the number of iterations, also note alpha is a function of num_iter
            num_iter += 1
            
            # Where g_x is the summation of all of our fi's at each x
            g_x = np.ndarray(shape=(self.d,1))
            g_x.fill(0)
            for fi in self.lambdas:
                g_x = np.add(g_x, fi(self.x))
            
            # Here we do the update if we aren't within the epsilon value
            if np.linalg.norm(g_x) > self.epsilon:
                # Note that fi is a function that goes from fi: R^d -> R^d, xk+1 = xk = alpha(num_iter)*f(xk)
                self.x = np.subtract(self.x, np.multiply(self.alpha(num_iter),g_x))
            else:
                cont_iter = False
                print "Gradient descent has found a solution:"
                print self.x
            
            # If we can't find a value within the max iteration limit return None
            if num_iter > self.max_iter:
                print "Stopping after reaching iteration limit."
                cont_iter = False