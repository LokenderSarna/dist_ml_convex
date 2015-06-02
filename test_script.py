from dist_ml_convex2 import *
import numpy as np
import inspect

Q1 = np.random.random((10,10))
Q1 = 0.5*(Q1 + np.transpose(Q1))
Q1 = np.linalg.matrix_power(Q1, 2)
Q2 = np.random.random((10,10))
Q2 = 0.5*(Q2 + np.transpose(Q2))
Q2 = np.linalg.matrix_power(Q2, 2)
Q3 = np.random.random((10,10))
Q3 = 0.5*(Q3 + np.transpose(Q3))
Q3 = np.linalg.matrix_power(Q3, 2)

q1 = np.random.random((10,1))
q2 = np.random.random((10,1))
q3 = np.random.random((10,1))

Q = Q1 + Q2 + Q3
q = q1 + q2 + q3

print "Closed form Solution: "
print - np.dot(np.linalg.inv(Q), q)

f1 = lambda x : (1/2)*(np.dot(np.dot(np.transpose(x), Q1), x)) + np.dot(q1, x)
f2 = lambda x : (1/2)*(np.dot(np.dot(np.transpose(x), Q2), x)) + np.dot(q2, x)
f3 = lambda x : (1/2)*(np.dot(np.dot(np.transpose(x), Q3), x)) + np.dot(q3, x)

grad_f1 = lambda x : np.dot(Q1, x) + q1
grad_f2 = lambda x : np.dot(Q2, x) + q2
grad_f3 = lambda x : np.dot(Q3, x) + q3

oracle1 = FirstOrderOracle(f1, grad_f1, 10) 
oracle2 = FirstOrderOracle(f2, grad_f2, 10) 
oracle3 = FirstOrderOracle(f3, grad_f3, 10) 

oracles = [oracle1, oracle2, oracle3]

grad = GradientDescent(oracles)
#grad.execute()
