from dist_ml_convex import *
import numpy as np

lambdas = [lambda x : 2*x +6]
prob = convex_problem(1, 1, lambdas, lambdas[0](2))

grad = gradient_descent(prob)
grad.execute()