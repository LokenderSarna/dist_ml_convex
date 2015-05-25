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
EPSILON_DEFAULT = 0.1
MAX_ITER_DEFAULT = 1000
ALPHA_DEFAULT = 0.01

# A method to find the p=2 norm of a set of lambda function
# NOTE: can we have the lambda take input an array?????
# Is this saying that each fi(x) has d dimensions?????
def l2norm(fi_x):
    
    if isinstance(fi_x, np.int64):
        return np.absolute(fi_x)

    if isinstance(fi_x, int):
        return np.absolute(fi_x)

    else:
        sum = 0
        for dimension in fi_x:
            sum += math.pow(dimension, 2)
        return math.sqrt(sum)

# Gradient_descent will take a convex problem as input and find the optimal x* 
class gradient_descent:
    
    # Parameters taken as input; alpha: the step size function, epsilon the stopping critera, epsilon: the stop criterion, max_iter: upper limit on the number of iterations
    def __init__(self, convex_problem, alpha=None, epsilon=None, max_iter=None):
        self.convex_problem = convex_problem
        self.d = convex_problem.d
        self.n = convex_problem.n
        self.lambdas = convex_problem.lambdas
        self.x = convex_problem.x0
        # Since alpha will be a function our default will also be a function to match type checking
        if alpha is None:
            alpha_def = np.zeros(self.d)
            alpha_def.fill(ALPHA_DEFAULT)
            self.alpha = (lambda x : alpha_def)
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
        
        #for each fi function we need to minimize such that the fi (gradient function) is within epsilon
        for fi in self.lambdas:
                        
            # Iterate through until the x vector values allow this fi to fall within epsilon
            inside_epsilon = False
            num_iter = 0
            while inside_epsilon == False:
                
                num_iter += 1
                # If we can't find a value within the max iteration limit return None
                if num_iter > self.max_iter:
                    return None

                # Here we do the update if we aren't within the epsilon value
                fi_x = fi(self.x)
                print l2norm(fi_x)
                if l2norm(fi_x) > self.epsilon:
                    # Note that fi is a function that goes from fi: R^d -> R^d, xk+1 = xk = alpha(xk)*f(xk)
                    self.x = self.x - self.alpha(self.x)*fi_x
                    print "grad @ fi_x: %f, x: %f" %(fi_x, self.x)
                else:
                    inside_epsilon = True
                    print "DONE!"
        
        
        