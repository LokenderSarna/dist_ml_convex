import numpy as np
import pandas as pd

# Defaults for gradient descent parameters
EPSILON_DEFAULT = 0.01
MAX_ITER_DEFAULT = 10000
ALPHA_DEFAULT = 0.001

# Single process linear regression will test to see if our dot product function is the problem still even in single process.
class LinearRegressionSingleProcess:

    # Just initializing the two needed paramaters: X and Y. Other paramaters like alpha and max iteration will be hard coded
    def __init__(self, X, y):
        self.y = np.matrix(y).T
        self.X = np.column_stack((X, np.ones(len(X))))
        
        # Initialize the weights vector w as random values, could have used zeros
        self.w = np.random.random( ( len(self.X[0]), 1 ) )
        # print "self.w"
        # print self.w
        # print "self.y"
        # print self.y
    
    # Method to return the weights w of each column vector will perform gradient descent
    def fit(self):
                
        cont_iter = True
        num_iter = 0
        
        # Now we are going to step through the possible w values until we find values within in our threshold
        while cont_iter == True:
            
            num_iter += 1;
            
            # If we can't find a value within the max iteration limit stop the execution
            if num_iter > MAX_ITER_DEFAULT:
                print "Stopping after reaching iteration limit."
                cont_iter = False
                continue
            
            # Find sum of each gradient grad_fi(x), since this is a single process version there is only one grad_f(w)
            grad_f = lambda x : 2*(np.dot(np.dot(self.X.T, self.X), x) - np.dot(self.X.T, self.y))
            grad_fw = grad_f(self.w)            
            
            # Check to see if our values are within the epsilon threshold, redundent to do norm with one value, but just for logic I will leave it
            if np.linalg.norm(grad_fw) > EPSILON_DEFAULT:
                # Note that fi is a function that goes from fi: R^d -> R^d, xk+1 = xk - alpha(num_iter)*f(xk)
                self.w = np.subtract(self.w, np.multiply(ALPHA_DEFAULT, grad_fw))
                # print self.w
            # If we don't need to update then we have found a solution, so just tell the loop to stop iterating
            else:
                print "Found solution for weights w:"
                print self.w
                cont_iter = False
    
    
    
    
        
    