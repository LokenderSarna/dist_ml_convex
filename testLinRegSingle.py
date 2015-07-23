from LinearRegressionSingleProcess import *
import numpy as np
import time
import pandas as pd
from sklearn import datasets, linear_model
"""
# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X_temp = diabetes_X[:, :, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-20]
diabetes_X_test = diabetes_X_temp[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Now we are going to compare to sklearns, Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)

lin = LinearRegressionSingleProcess(diabetes_X_train, diabetes_y_train)
lin.fit()
"""


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

# Now we are going to compare to sklearns, Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train.as_matrix(), Y_train.as_matrix())
# The coefficients
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)

lin = LinearRegressionSingleProcess( X_train.as_matrix(), Y_train.as_matrix() )
lin.fit()


