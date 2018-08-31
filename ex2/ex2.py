#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 21:22:57 2018

@author: varshabhat
"""

#%% Machine Learning Online Class - Exercise 2: Logistic Regression
#%
#%  Instructions
#%  ------------
#% 
#%  This file contains code that helps you get started on the logistic
#%  regression exercise. You will need to complete the following functions 
#%  in this exericse:
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
#%% Initialization
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

data_2 = np.loadtxt('ex2data1.txt',delimiter=',')
X_shape = np.shape(data_2)

X = np.reshape(data_2[:,0:X_shape[1]-1],(len(data_2),X_shape[1]-1))
y = np.reshape(data_2[:,X_shape[1]-1],(len(data_2),1))
#
# ==================== Part 1: Plotting ====================

plotData(X,y)
plt.clf()

#%% ============ Part 2: Compute Cost and Gradient ============
#%  In this part of the exercise, you will implement the cost and gradient
#%  for logistic regression. You neeed to complete the code in 
#%  costFunction.m
#
#%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = np.shape(X)

#
#% Add intercept term to x and X_test
X = np.append(np.ones((len(X),1)),X, axis = 1)
#
#% Initialize fitting parameters
initial_theta = np.zeros((n+1, 1))


# Compute and display initial cost and gradient
[cost, grad] = [costFunction(initial_theta, X, y),gradient(initial_theta, X, y)]
#
print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): {}'.format(grad))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');
#
#% Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
[cost, grad] = [costFunction(test_theta, X, y),gradient(test_theta, X, y)]

print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.218\n');
print('Gradient at initial theta (zeros): {}'.format(grad))
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

#%% ============= Part 3: Optimizing using fminunc  =============
#%  In this exercise, you will use a built-in function (fminunc) to find the
#%  optimal parameters theta.
#
#%  Set options for fminunc
#options = optimset('GradObj', 'on', 'MaxIter', 400);
#
#%  Run fminunc to obtain the optimal theta
#%  This function will return theta and the cost 

result = opt.fmin_tnc(func = costFunction, x0 = test_theta,fprime = gradient, args = (X, y))
cost = costFunction(result[0], X, y)

#% Print theta to screen
print('Cost at theta found by fminunc:{}'.format(cost))
print('Expected cost (approx): 0.203\n')
print('theta: {}'.format(theta))
print(' %f \n'.format(result[0]))
print('Expected theta (approx):\n  -25.161\n 0.206\n 0.201\n')
theta = result[0]
#% Plot Boundary
#plotDecisionBoundary(theta, X, y);
#
#% Put some labels 
#hold on;
#% Labels and Legend
#xlabel('Exam 1 score')
#ylabel('Exam 2 score')
#
#% Specified in plot order
#legend('Admitted', 'Not admitted')
#hold off;
#
#fprintf('\nProgram paused. Press enter to continue.\n');
#pause;
#
#%% ============== Part 4: Predict and Accuracies ==============
#%  After learning the parameters, you'll like to use it to predict the outcomes
#%  on unseen data. In this part, you will use the logistic regression model
#%  to predict the probability that a student with score 45 on exam 1 and 
#%  score 85 on exam 2 will be admitted.
#%
#%  Furthermore, you will compute the training and test set accuracies of 
#%  our model.
#%
#%  Your task is to complete the code in predict.m
#geldl;;   
X = [ 1, 45, 85]
#%  Predict probability for a student with score 45 on exam 1 
#%  and score 85 on exam 2 
#
prob = sigmoid(X @ theta)
print(prob)
print('For a student with scores 45 and 85, we predict an admission probability of {}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n\n')
#
#% Compute accuracy on our training set
p = predict(theta, X)




