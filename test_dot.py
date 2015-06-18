from dist_ml_convex import *
import numpy as np
import time
import pandas as pd

# Gather data from train and test csv 
data_train = pd.read_csv('../train.csv', skipinitialspace=1, index_col=0, parse_dates=True)

# Explode datetime into hour, dayofweek, and month
data_train['hour'] = data_train.index.hour
data_train['dayofweek'] = data_train.index.weekday
data_train['month'] = data_train.index.month

# Build a dataset with its matching y outputs
X_selected_cols = [u'weather',u'dayofweek',u'hour',u'season',u'holiday',u'workingday',u'temp',u'atemp',u'humidity',u'windspeed']
X_train = data_train[X_selected_cols]
Y_train = data_train['count']

# Now split each data matrix needed into a flattened array for broadcasting
X1 = X_train._slice(slice(0,2500)).as_matrix()
Y1 = Y_train._slice(slice(0,2500)).as_matrix()
X2 = X_train._slice(slice(2500,5000)).as_matrix()
Y2 = Y_train._slice(slice(2500,5000)).as_matrix()
X3 = X_train._slice(slice(5000,7500)).as_matrix()
Y3 = Y_train._slice(slice(5000,7500)).as_matrix()
X4 = X_train._slice(slice(7500,10886)).as_matrix()
Y4 = Y_train._slice(slice(7500,10886)).as_matrix()

X = X1
Y = Y1.reshape((2500,1))

w = np.random.random((10, 1))
# print X.shape
# print w.shape
# Xw = np.dot(X, w)
# print Xw.shape
# XT = X.T
# print XT.shape
# XTXw = np.dot(XT, Xw)
# print XTXw.shape

print "Starting calculation."
t0 = time.time()
XT = X.T
# XTX = XT.dot(X)
# XTXw = XTX.dot(w)
# XTY = XT.dot(Y)
# wf = XTXw - XTY
# print wf

wf = np.dot(np.dot(XT, X), w) - np.dot(XT, Y)
print wf

t1 = time.time()


# print ( np.dot(np.dot(X.T, X), w) - np.dot(X.T, Y) ).shape
#print XTXw.shape
#print XTY.T.shape
#print wf.shape
print "Calculation finished. (%f seconds)" %(t1-t0)

