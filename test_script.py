from dist_ml_convex import *
import numpy as np

#lambdas = [lambda x : 2*x +6]
#prob = ConvexProblem(1, 1, lambdas, lambdas[0](2))

Q1 = np.random.random((5,5))
Q1 = 0.5*(Q1 + np.transpose(Q1))
Q1 = np.linalg.matrix_power(Q1, 2)
Q2 = np.random.random((5,5))
Q2 = 0.5*(Q2 + np.transpose(Q2))
Q2 = np.linalg.matrix_power(Q2, 2)
Q3 = np.random.random((5,5))
Q3 = 0.5*(Q3 + np.transpose(Q3))
Q3 = np.linalg.matrix_power(Q3, 2)

q1 = np.random.random((5,1))
q2 = np.random.random((5,1))
q3 = np.random.random((5,1))

lambdas = [ lambda x : np.dot(Q1, x) + q1, lambda x : np.dot(Q2, x) + q2, lambda x : np.dot(Q3, x) + q3 ]

prob = ConvexProblem(5, lambdas)

grad = GradientDescent(prob)

Q = Q1 + Q2 + Q3
q = q1 + q2 + q3

grad.execute()

print "Closed form Solution: "
print - np.dot(np.linalg.inv(Q), q)