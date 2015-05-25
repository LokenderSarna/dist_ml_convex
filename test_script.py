from dist_ml_convex import *

lambdas = [lambda x : x + 2, lambda x : x + 5]
prob = convex_problem(3, 2, lambdas, 4)

grad = gradient_descent(prob)
grad.execute()