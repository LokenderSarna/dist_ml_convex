from dist_ml_convex3 import *
import numpy as np

dim_f = 3

Q1 = np.array([[1.10940746, 0.84856369, 0.96808427], [0.84856369, 0.71776606, 0.81411964], [0.96808427, 0.81411964, 1.26411601]])
Q2 = np.array([[0.80520088, 0.62023123, 0.63628587], [0.62023123, 0.72322862, 0.71080702], [0.63628587, 0.71080702, 0.92083444]])
Q3 = np.array([[1.5451591, 1.18724693, 1.06231687], [1.18724693, 0.96792878, 0.80150356], [1.06231687, 0.80150356, 0.78486679]])

q1 = np.array([[0.15073246], [ 0.70496296], [ 0.90001029]])
q2 = np.array([[ 0.84843654], [ 0.95739634], [ 0.23685848]])
q3 = np.array([[ 0.63475301], [ 0.92054327], [ 0.79867474]])

# Q1 = np.random.random((dim_f,dim_f))
# Q1 = 0.5*(Q1 + np.transpose(Q1))
# Q1 = np.linalg.matrix_power(Q1, 2)
# Q2 = np.random.random((dim_f,dim_f))
# Q2 = 0.5*(Q2 + np.transpose(Q2))
# Q2 = np.linalg.matrix_power(Q2, 2)
# Q3 = np.random.random((dim_f,dim_f))
# Q3 = 0.5*(Q3 + np.transpose(Q3))
# Q3 = np.linalg.matrix_power(Q3, 2)

# q1 = np.random.random((dim_f,1))
# q2 = np.random.random((dim_f,1))
# q3 = np.random.random((dim_f,1))



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

oracle1 = FirstOrderOracle(f1, grad_f1, dim_f) 
oracle2 = FirstOrderOracle(f2, grad_f2, dim_f) 
oracle3 = FirstOrderOracle(f3, grad_f3, dim_f) 

oracles = [oracle1, oracle2, oracle3]

grad = GradientDescent(oracles)

grad.execute()