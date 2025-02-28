# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 01:04:07 2025

@author: kylek
"""

import numpy as np
import matplotlib.pyplot as plt

#%% problem 4 (1)
print("\nPromblem 4\n")

A = np.array([[2,-1,3,4,-5,6,7,8],
             [1,2,4,3,6,5,8,7],
             [3,4,2.6,1,0,-7,6.9,5],
             [4,3,1,0,7,8,5,6],
             [-5,6,8,7,1,2.5,4,3],
             [6,5,7,8,2,1.1,3,4],
             [7,-8,0,6,4,3,0,2],
             [8,7,6,5,3,4,2,1]])


B = np.array([[0,3,0,4,5,-7,-6,8],
              [3,-1,4,2,7,-5.5,-8,6],
              [2,4.4,1,3,6.2,8,5,7],
              [4,2,3,1,8,6,7,5],
              [5,7.1,6,8,1,0,-2,4],
              [7,5,-8,6,3,1,-4.7,2],
              [0,8.8,5,0,2,4,1,3],
              [8,6,7,5,4,0,3,1]])

print(np.linalg.det(np.linalg.matmul(A, B)))
print ("\n\n")

#%% problem 5 (2)

print("problem 5\n")

C = np.array([[4,3,2,1,5,6,7,8],
              [3,4,1,2,7,8,5,6],
              [2,1,4,3,6,5,8,7],
              [1,2,3,4,8,7,6,5],
              [5,7,6,8,4,3,2,1],
              [6,5,8,7,3,4,1,2],
              [7,8,5,6,2,1,4,3],
              [8,6,7,4,1,2,3,4]])

print(np.linalg.inv(C))
print("\n\n")

#%% problem 6 (3)

print("problem 6\n")

x = True

D = np.array([[0.5,0.25,0.25],
              [0.2,0.7,0.1],
              [0.3,0.3,0.4]])

while x == True:
    
    Dsum = np.sum(D)
    
    D = np.linalg.matmul(D, D)
    
    if (np.abs(Dsum-np.sum(D)) < 0.001):
        x = False
        
print(D)
print("\n delta sum = ")
print(np.abs(Dsum-np.sum(D)))
print("\n\n")
    
#%% harmonic oscillators
print("Harmonic Oscillators\n")

M = np.array([[1,0,0,0,0],
              [0,2,0,0,0],
              [0,0,1,0,0],
              [0,0,0,2,0],
              [0,0,0,0,3]])

k1 = 3
k2 = 5
k3 = 4
k4 = 6
k5 = 2
k6 = 7

K = np.array([[k1+k2,-k2,0,0,0],
              [-k2,k2+k3,-k3,0,0],
              [0,-k3,k3+k4,-k4,0],
              [0,0,-k4,k4+k5,-k5],
              [0,0,0,-k5,k5+k6]])

evals, evecs = np.linalg.eig(np.linalg.matmul(K,np.linalg.inv(M)))
#evals, evecs = np.linalg.eig(K)
print(evals)
print('\n')
print(evecs)



