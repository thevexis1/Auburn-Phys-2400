#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:25:45 2025

@author: bradleyandrew
"""

import numpy as np

# --- Problem 1: Matrix-Vector Multiplication ---
A = np.array([[2, -1, 3],
              [1, 0, -2],
              [4, 1, 5]])  # 3x3 matrix

v = np.array([1, -1, 2])  # 3x1 vector

result = np.dot(A, v)  # Matrix-vector multiplication

print("Problem 1: Matrix-Vector Multiplication")
print(f"A * v =\n{result}\n")


# --- Problem 2: Determinant of a 3x3 Matrix ---
B = np.array([[4, 0, -2],
              [1, 3, 5],
              [7, -1, 6]])  # Matrix with a zero in top middle

det_B = np.linalg.det(B)  # Compute determinant

print("Problem 2: Determinant Calculation")
print(f"det(B) = {det_B:.2f}\n")


# --- Problem 3: Solving a System of Equations ---
# System: Ax = b, where A is the coefficient matrix, and b is the right-hand side vector
C = np.array([[3, -2, 1],
              [1, 4, -3],
              [5, 1, 2]])  # Coefficients of xi, eta, zeta

b = np.array([4, -2, 7])  # Right-hand side values

solution = np.linalg.solve(C, b)  # Solve for [xi, eta, zeta]

print("Problem 3: Solving System of Equations")
print(f"xi = {solution[0]:.2f}, eta = {solution[1]:.2f}, zeta = {solution[2]:.2f}")

#%%

"""

This section will be for you to solve the next two sections of the assignment

"""
A = np.array([[2,-1,3,4,-5,6,7,8],
              [1,2,4,3,6,5,8,7],
             [3,4,2.6,1,0,-7,6.9,5],
             [4,3,1,0,7,8,5,6],
             [-5,6,8,7,1,2.5,4,3],
             [6,5,7,8,2,1.1,3,4],
             [7,-8,0,6,4,3,0,2],
             [8,7,6,5,3,4,2,1]])

# To compute the inverse ex: C_inv = np.linalg.inv(C)