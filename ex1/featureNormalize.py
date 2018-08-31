#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 21:50:23 2018

@author: varshabhat
"""

def featureNormalize(X):
    mu = np.mean(X)
    sigma = np.std(X)
    X_norm =(X - mu*np.ones(size(X,1),size(X,2)))/(sigma*np.ones(size(X,1),size(X,2)))