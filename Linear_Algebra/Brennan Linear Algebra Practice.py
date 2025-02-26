#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:30:31 2025

@author: brennan
"""
import numpy as np
'Question 1' 
#Matrix
A = np.array([[2,-1,3,4,-5,6,7,8],
              [1,2,4,3,6,5,8,7],
             [3,4,2.6,1,0,-7,6.9,5],
             [4,3,1,0,7,8,5,6],
             [-5,6,8,7,1,2.5,4,3],
             [6,5,7,8,2,1.1,3,4],
             [7,-8,0,6,4,3,0,2],
             [8,7,6,5,3,4,2,1]])

B = np.array ([[0,3,0,4,5,-7,-6,8],
               [3,-1,4,2,7,-5.5,-8,6],
               [2,4.4,1,3,6.2,8,5,7],
               [4,2,3,1,8,6,7,5],
               [5,7.1,6,8,1,0,-2,4],
               [7,5,-8,6,3,1,-4.7,2],
               [0,8.8,5,0,2,4,1,3],
              [8,6,7,5,4,0,3,1]])

#Compute Matrix Product C = A*B
C = np.matmul(A,B)

#Compute Determinate of C
det_C = np.linalg.det(C)

print('det(A x B) =', det_C)
#%%
'Question 2' 
#Matrix
C = ([[4,3,2,1,5,6,7,8],
     [3,4,1,2,7,8,5,6],
     [2,1,4,3,6,5,8,7],
     [1,2,3,4,8,7,6,5],
     [5,7,6,8,4,3,2,1],
     [6,5,8,7,3,4,1,2],
     [7,8,5,6,2,1,4,3],
     [8,6,7,5,1,2,3,4]])
#Compute Inverse 
C_inv = np.linalg.inv(C)
print("inv(C) = ", C_inv)
#%%
'Question 3'
'Told ChatGPT to make a code in python that multiplies d by itself until the resulting matrix stops changing significantly (i.e. converges to a steady-state matrix). Determine how many iterations it takes for the matrix to reach steady values.'
#Matrix
D = ([[0.5,0.25,0.25],
      [0.2,0.7,0.1],
      [0.3,0.3,0.4]])

#Set a convergence threshold
threshold = 1e-6
max_iterations = 1000 #To prevent infinite loops 
iteration = 0

#Start with original matrix
D_prev = D.copy()

while iteration < max_iterations: 
    iteration +=1
    D_new = np.matmul(D_prev, D) #Multiply D by itself
    
    #Check the maximum absolute diffrence between elements 
    if np.max(np.abs(D_new - D_prev)) < threshold:
        break #Stop if changes are smaller than the threshold
        
    D_prev = D_new #Update for the next iteration 
        
#Output results 
print(f"Steady-state matrix reached in {iteration} iterations:\n", D_new)