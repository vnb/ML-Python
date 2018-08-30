#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 00:34:13 2018

@author: varshabhat
"""

#function J = computeCost(X, y, theta)
#%COMPUTECOST Compute cost for linear regression
#%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#%   parameter for linear regression to fit the data points in X and y
#
#% Initialize some useful values
#m = length(y); % number of training examples
#
#% You need to return the following variables correctly 
#J = 0;
#
#% ====================== YOUR CODE HERE ======================
#% Instructions: Compute the cost of a particular choice of theta
#%               You should set J to the cost.
#h = sum(X*theta,2);
#J = sum(0.5*1/m*(h- y).^2);
#
#
#% =========================================================================
#
#end
import numpy as np
import pandas as pd
def computeCost(X,y,theta):
    m = len(X)
    h = X@theta
    J = np.sum(0.5/m*(h-y)**2)
    return h,J


#function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
#%GRADIENTDESCENT Performs gradient descent to learn theta
#%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#%   taking num_iters gradient steps with learning rate alpha
#
#% Initialize some useful values
#m = length(y); % number of training examples
#J_history = zeros(num_iters, 1);
#
#for iter = 1:num_iters
#
#    % ====================== YOUR CODE HERE ======================
#    % Instructions: Perform a single gradient step on the parameter vector
#    %               theta. 
#    %
#    % Hint: While debugging, it can be useful to print out the values
#    %       of the cost function (computeCost) and gradient here.
#    %
#    h = sum(X*theta,2);
#    theta = theta - X'.*alpha * 1/m * (h-y);                                                                                                                                                                                                                          
#    % ============================================================
#
#    % Save the cost J in every iteration    
#    J_history(iter) = computeCost(X, y, theta);
#end
#end

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(X)
    for i in range(num_iters):
        h = X@theta
        diff = (h-y)*alpha/m
        X_t = np.transpose(X)
        theta = theta - X_t@diff
    return theta
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    