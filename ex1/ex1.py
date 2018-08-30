# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#% Machine Learning Online Class - Exercise 1: Linear Regression
#
#%  Instructions
#%  ------------
#%
#%  This file contains code that helps you get started on the
#%  linear exercise. You will need to complete the following functions
#%  in this exericse:
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
#%  For this exercise, you will not need to change any code in this file,
#%  or any other files other than those mentioned above.
#%
#% x refers to the population size in 10,000s
#% y refers to the profit in $10,000s
#%
#
#%% Initialization
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import numpy as np
import pandas as pd
from ex1_funcs import computeCost
from ex1_funcs import gradientDescent
import matplotlib.pyplot as plt

#%% ==================== Part 1: Basic Function ====================

print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
data_1 = np.loadtxt('ex1data1.txt',delimiter=',')
# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
#fprintf('Plotting Data ...\n')

X = np.reshape(data_1[:,0],(len(data_1),1))
y = np.reshape(data_1[:,1],(len(data_1),1))
m = len(X)




plt.figure(1)
plt.plot(X,y,'bo', label = 'Training Data')
plt.legend()
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
plt.title('Linear Regression - Profit vs Population')
#plt.show()

#=================== Part 3: Cost and Gradient descent ===================

X = np.append(np.ones((len(X),1)),X, axis = 1)
theta = np.zeros((2,1))
#Some gradient descent settings
iterations = 1500
alpha = 0.01


print('\nTesting the cost function ...\n')
#compute and display initial cost
cost = computeCost(X, y, theta)
print('Cost computed = {}'.format(cost[1]))
print('Expected cost value (approx) 32.07\n')

theta = np.array([[-1],[2]])

cost = computeCost(X, y, theta)
print('Cost computed = {}'.format(cost[1]))
print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('{}'.format(theta))
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
ax.plot(X[:,1], X@theta, 'r-',label = 'Linear Regression')
plt.figure(1)
plt.plot(X[:,1], X@theta, 'r-',label = 'Linear Regression')
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
plt.title('Linear Regression - Profit vs Population')
plt.legend(loc = 'upper right')
#plt.legend('a','b')
plt.show()

#============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')
#
# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, num = 100)
theta1_vals = np.linspace(-1, 4, num = 100)
#
# initialize J_vals to a matrix of 0's
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])
#


# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = [theta0_vals[i], theta1_vals[j]]
        J_vals[i][j] = computeCost(X, y, t)[1]

#
#% Because of the way meshgrids work in the surf command, we need to
#% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = np.transpose(J_vals)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = theta0_vals
Y = theta1_vals
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = J_vals

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#
#plt.show()
#
#space = np.logspace(-2,3,20)
#
#plt.contour(theta0_vals,theta1_vals,Z, 50)
fig, ax = plt.subplots()
cs = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)

plt.show()