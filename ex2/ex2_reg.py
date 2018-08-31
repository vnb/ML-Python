#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 13:17:51 2018

@author: varshabhat
"""

#%% Machine Learning Online Class - Exercise 2: Logistic Regression
#%
#%  Instructions
#%  ------------
#%
#%  This file contains code that helps you get started on the second part
#%  of the exercise which covers regularization with logistic regression.
#%
#%  You will need to complete the following functions in this exericse:
#%
#%     sigmoid.m
#%     costFunction.m
#%     predict.m
#%     costFunctionReg.m
#%
#%  For this exercise, you will not need to change any code in this file,
#%  or any other files other than those mentioned above.
#%
#
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import scipy.optimize as opt  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ex2_funcs import *

#%  The first two columns contains the exam scores and the third column
#%  contains the label.

data_2 = np.loadtxt('ex2data2.txt',delimiter=',')
X_shape = np.shape(data_2)

X = np.reshape(data_2[:,0:X_shape[1]-1],(len(data_2),X_shape[1]-1))
y = np.reshape(data_2[:,X_shape[1]-1],(len(data_2),1))
#
# ==================== Part 1: Plotting ====================

plotData2(X,y)
plt.clf()

#% Specified in plot order
#legend('y = 1', 'y = 0')
#hold off;
#
#
#%% =========== Part 1: Regularized Logistic Regression ============
#%  In this part, you are given a dataset with data points that are not
#%  linearly separable. However, you would still like to use logistic
#%  regression to classify the data points.
#%
#%  To do so, you introduce more features to use -- in particular, you add
#%  polynomial features to our data matrix (similar to polynomial
#%  regression).
#%
#
#% Add Polynomial Features
#
#% Note that mapFeature also adds a column of ones for us, so the intercept
#% term is handled
X1 = np.reshape(X[:,0],(np.shape(X)[0],1))
X2 = np.reshape(X[:,1],(np.shape(X)[0],1))
X = mapFeature(X1, X2)

#
#% Initialize fitting parameters
initial_theta = np.zeros((np.shape(X)[1], 1))
#
#% Set regularization parameter lambda to 1
lambda_r = 1
#% Compute and display initial cost and gradient for regularized logistic
#% regression
cost = costFunctionReg(initial_theta, X, y, lambda_r)[0]
grad = costFunctionReg(initial_theta, X, y, lambda_r)[1]

print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n');
print(' {}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')


#% Compute and display cost and gradient
#% with all-ones theta and lambda = 10
test_theta = np.ones((np.shape(X)[1], 1))
cost = costFunctionReg(test_theta, X, y, 10)[0]
grad = costFunctionReg(test_theta, X, y, 10)[1]

print('Cost at test theta (zeros): {}'.format(cost))
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(' {}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

#TO DO#
#%% ============= Part 2: Regularization and Accuracies =============
#%  Optional Exercise:
#%  In this part, you will get to try different values of lambda and
#%  see how regularization affects the decision coundart
#%
#%  Try the following values of lambda (0, 1, 10, 100).
#%
#%  How does the decision boundary change when you vary lambda? How does
#%  the training set accuracy vary?
#%
#
#% Initialize fitting parameters
#initial_theta = zeros(size(X, 2), 1);
#
#% Set regularization parameter lambda to 1 (you should vary this)
#lambda = 1;
#
#% Set Options
#options = optimset('GradObj', 'on', 'MaxIter', 400);
#
#% Optimize
#[theta, J, exit_flag] = ...
#	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
#
#% Plot Boundary
#plotDecisionBoundary(theta, X, y);
#hold on;
#title(sprintf('lambda = %g', lambda))
#
#% Labels and Legend
#xlabel('Microchip Test 1')
#ylabel('Microchip Test 2')
#
#legend('y = 1', 'y = 0', 'Decision boundary')
#hold off;
#
#% Compute accuracy on our training set
#p = predict(theta, X);
#
#fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
#fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');