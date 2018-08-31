##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Wed Aug 29 21:29:52 2018
#
#@author: varshabhat
#"""
#
#%% Machine Learning Online Class
#%  Exercise 1: Linear regression with multiple variables
#%
#%  Instructions
#%  ------------
#% 
#%  This file contains code that helps you get started on the
#%  linear regression exercise. 
#%
#%  You will need to complete the following functions in this 
#%  exericse:
#%
#%     warmUpExercise.m
#%     plotData.m
#%     gradientDescent.m
#%     computeCost.m
#%     gradientDescentMulti.m
#%     computeCostMulti.m
#%     featureNormalize.m
#%     normalEqn.m
#%
#%  For this part of the exercise, you will need to change some
#%  parts of the code below for various experiments (e.g., changing
#%  learning rates).
#%
#
#%% Initialization
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import sklearn.preprocessing
import numpy as np
import pandas as pd
from ex1_funcs import computeCost
from ex1_funcs import gradientDescent
import matplotlib.pyplot as plt
#%% ================ Part 1: Feature Normalization ================
#%% Load Data
data_2 = np.loadtxt('ex1data2.txt',delimiter=',')
X_shape = np.shape(data_2)

X = np.reshape(data_2[:,0:X_shape[1]-1],(len(data_2),X_shape[1]-1))
y = np.reshape(data_2[:,X_shape[1]-1],(len(data_2),1))

def featureNormalize(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis = 0)
    d = np.ones((X_shape[0],X_shape[1]-1))
    X_norm =(X - mu*d)/(sigma*d)
    return X_norm,mu,sigma

# Scale features and set them to zero mean
print('Normalizing Features ...\n')
#print(sklearn.preprocessing.normalize(X))
[X, mu, sigma] = featureNormalize(X)
# Add intercept term to X

X = np.append(np.ones((len(X),1)),X, axis = 1)


#
#%% ================ Part 2: Gradient Descent ================
#
#% ====================== YOUR CODE HERE ======================
#% Instructions: We have provided you with the following starter
#%               code that runs gradient descent with a particular
#%               learning rate (alpha). 
#%
#%               Your task is to first make sure that your functions - 
#%               computeCost and gradientDescent already work with 
#%               this starter code and support multiple variables.
#%
#%               After that, try running gradient descent with 
#%               different values of alpha and see which one gives
#%               you the best result.
#%
#%               Finally, you should complete the code at the end
#%               to predict the price of a 1650 sq-ft, 3 br house.
#%
#% Hint: By using the 'hold on' command, you can plot multiple
#%       graphs on the same figure.
#%
#% Hint: At prediction, make sure you do the same feature normalization.
#%

def computeCostMulti(X,y,theta):
    m = len(X)
    h = X@theta
    J = np.sum(0.5/m*(h- y)**2)
    return h,J
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y) #% number of training examples
    J_history = np.zeros([num_iters, 1])

    for i in range(num_iters):
        h = computeCostMulti(X,y,theta)[0]
        theta = theta - alpha/m*(np.transpose(X) @ (h-y))

#     Save the cost J in every iteration    
        J_history[i] = computeCostMulti(X, y, theta)[1]
    return theta,J_history





print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.001
num_iters = 4000

# Init Theta and Run Gradient Descent 
theta = np.zeros([X_shape[1], 1])
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)


# Plot the convergence graph

plt.plot(range(num_iters),J_history, 'b-', linewidth = 2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
#% Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(' {} \n'.format(theta))


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = [1, (1650 - mu[0])/sigma[0], (3 - mu[1]/sigma[1])]@theta
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):{}\n'.format(price[0]))



##% ================ Part 3: Normal Equations ================
##
##fprintf('Solving with normal equations...\n');
##
##% ====================== YOUR CODE HERE ======================
##% Instructions: The following code computes the closed form 
##%               solution for linear regression using the normal
##%               equations. You should complete the code in 
##%               normalEqn.m
##%
##%               After doing so, you should complete this code 
##%               to predict the price of a 1650 sq-ft, 3 br house.
##%
##
##%% Load Data
data_2 = np.loadtxt('ex1data2.txt',delimiter=',')
X_shape = np.shape(data_2)
X = np.reshape(data_2[:,0:X_shape[1]-1],(len(data_2),X_shape[1]-1))
y = np.reshape(data_2[:,X_shape[1]-1],(len(data_2),1))
m = len(y)

## Add intercept term to X
X = np.append(np.ones((len(X),1)),X, axis = 1)


def normalEqn(X,y):
    theta = (np.transpose(X)@X)**-1@np.transpose(X)@y
    return theta


##% Calculate the parameters from the normal equation
theta = normalEqn(X, y);
##
##% Display normal equation's result
print('Theta computed from the normal equations: \n')
print(' {}'.format(theta))
print('\n')
##
##
##% Estimate the price of a 1650 sq-ft, 3 br house
##% ====================== YOUR CODE HERE ======================
price = np.array([1, 1650, 3])@theta #% You should change this
print('Predicted price of a 1650 sq-ft, 3 br house \n (using normal equations):{}'.format(price[0]))


data_2 = np.loadtxt('ex1data2.txt',delimiter=',')
X_shape = np.shape(data_2)
X = np.reshape(data_2[:,0:X_shape[1]-1],(len(data_2),X_shape[1]-1))
y = np.reshape(data_2[:,X_shape[1]-1],(len(data_2),1))
m = len(y)

regr = linear_model.LinearRegression()
regr.fit(X, y)
h = regr.predict(X)
plt.scatter(h, y,  color='black')
#plt.plot(X, h, color='red', linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.show()
