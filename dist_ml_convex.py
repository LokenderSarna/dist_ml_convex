""" defaults for convex_problem """
x0_DEFAULT = 0

""" Here we define the class convex_problem to build a convex optimization problem to be sent to gradient descent """
class convex_problem:
	
    """ The problem contains a few basic parameters. 
    The number of dimensions in the vector x,
    The number of terms to be summed up in the cost function, the number of lambdas,
    The actual lambda functions which are the derivatives of each fi cost function term,
    The vector length of lambdas input and output: d,
    x0 the optional starting place for the problem to start, otherwise its set to 0 """
    def __init__(self, x_vector_dimensions, num_cost_function_terms, lambdas, lambdas_vector_length, x0=None):
        self.x_vector_dimensions = x_vector_dimensions
        self.num_cost_function_terms = num_cost_function_terms
        self.lambdas = lambdas
        self.lambdas_vector_length = lambdas_vector_length
        if x0 is None:
            self.x0 = x0_DEFAULT
        else:
            self.x0 = x0
    
    """ Simple method to print information on the convex problem """
    def print_info(self):
        print "Number of x vector dimensions = %d" %(self.x_vector_dimensions)
        print "Number of cost function terms = %d" %(self.num_cost_function_terms)
        print "Number of lambda functions = %d" %(len(self.lambdas))
        print "Length of lambdas input and output vector = %d" %(self.lambdas_vector_length)


""" Defaults for gradient descent parameters """
STEP_SIZE_DEFAULT = 0.5
STOPPING_CRITERION_DEFAULT = 0.01
MAX_ITERATIONS_DEFAULT = 100000

""" gradient_descent will take a convex problem as input and find the optimal x* """
class gradient_descent:
    
    """ The parameters sent to gradient descent include
    The actual convex problem to be optimized,
    The step size of the gradient descent algorithm,
    The stopping criterion (or epsilon) to say how specific the grid we are looking for is,
    The maximimum number of iteration before the algorithm gives up """
    def __init__(self, convex_problem, step_size=None, stopping_criterion=None, max_iterations=None):
        self.convex_problem = convex_problem
        if step_size is None:
            self.step_size = STEP_SIZE_DEFAULT
        else:
            self.step_size = step_size
        if stopping_criterion is None:
            self.stopping_criterion = STOPPING_CRITERION_DEFAULT
        else:
            self.stopping_criterion = stopping_criterion
        if max_iterations is None:
            self.max_iterations = MAX_ITERATIONS_DEFAULT
        else:
            self.max_iterations = max_iterations
            
    """ Simple method to print information on the gradient descent parameters (includes convex problem info) """
    def print_info(self):
        self.convex_problem.print_info()
        print "Step size of the descent algorithm (alpha) = %f" %(self.step_size)
        print "Stopping criterion of th algorithm = %f" %(self.stopping_criterion)
        print "Maximum number of iterations = %d" %(self.max_iterations)
            
            
            
            
