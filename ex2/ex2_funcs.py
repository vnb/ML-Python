#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 17:15:00 2018

@author: varshabhat
"""
# ====================== YOUR CODE HERE ======================
# Instructions: Plot the positive and negative examples on a
#               2D plot, using the option 'k+' for the positive
#               examples and 'ko' for the negative examples.

import matplotlib.pyplot as plt
import numpy as np
def plotData(X,Y):
    pos = np.where(Y == 1)[0]
    neg = np.where(Y == 0)[0]
    pos = np.reshape(pos, (len(pos),1))
    neg = np.reshape(neg, (len(neg),1))
    plt.plot(X[:,0][pos],X[:,1][pos],'b+',label = 'Positive')
    plt.plot(X[:,0][neg],X[:,1][neg], 'ro', label = 'Negative')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.show()
    
    
def plotData2(X,Y):
    pos = np.where(Y == 1)[0]
    neg = np.where(Y == 0)[0]
    pos = np.reshape(pos, (len(pos),1))
    neg = np.reshape(neg, (len(neg),1))
    plt.plot(X[:,0][pos],X[:,1][pos],'b+',label = 'Positive')
    plt.plot(X[:,0][neg],X[:,1][neg], 'ro', label = 'Negative')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.show()   
    
def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def costFunction(theta,X,y):
    m = len(X)
    np.reshape(theta,(3,1))
    h = sigmoid(X@theta)
    J = 1/m*(-np.transpose(y)@np.log(h) - np.transpose(1 - y)@np.log(1 - h))
    return J

def gradient(theta, X, y):
    m = len(X)
    grad = 1/m*np.transpose(sigmoid(X@theta) - y)@X
    return grad[0]
    

def predict(theta, X):
    m = len(X)
    h = sigmoid(X@theta)
    res = np.zeros((m,1))
    if np.where(h>=0.5):
        return 1
    else:
        return 0

def mapFeature(X1,X2):
    degree = 6
    out = np.ones((np.shape(X1)[0],1))
    for i in range(1,degree+1,1):
        for j in range(0,i+1,1):
            add_arr = (X1**(i-j)*X2**j)
            out = np.append(out,add_arr,axis = 1)
    return out
    


def costFunctionReg(theta, X, y, lambda_r):
    m = len(X)
    h = sigmoid(X@theta)
    theta_reg = theta[1:len(theta)]
    theta_reg = np.insert(theta_reg,0,0,axis=0)
    l_term = (lambda_r/(2*m))*np.transpose(theta_reg)@theta_reg
    J = 1/m*(-np.transpose(y)@np.log(h) - np.transpose(1 - y)@np.log(1 - h))+l_term
    grad = (1/m)*(np.transpose(X)@(h-y)+lambda_r*theta_reg)
    return J,grad

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    