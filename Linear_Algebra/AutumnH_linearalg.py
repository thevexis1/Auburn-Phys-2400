#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:25:45 2025

@author: lanay902
"""

import numpy as np

# --- Problem 1: Matrix-Vector Multiplication ---
A = np.array([[2, -1, 3],
              [0, 4, 1],
              [-2, 5, -3]])  # 3x3 matrix

v = np.array([1, 2, -1])  # 3x1 vector

result = np.dot(A, v)  # Matrix-vector multiplication

print("Problem 1: Matrix-Vector Multiplication")
print(f"A * v =\n{result}\n")


# --- Problem 2: Determinant of a 3x3 Matrix ---
B = np.array([[3, 0, 1],
              [-2, 4, 5],
              [1, -3, 2]])  # Matrix with a zero in top middle

det_B = np.linalg.det(B)  # Compute determinant

print("Problem 2: Determinant Calculation")
print(f"det(B) = {det_B:.2f}\n")


# --- Problem 3: Solving a System of Equations ---
# System: Ax = b, where A is the coefficient matrix, and b is the right-hand side vector
C = np.array([[2, -1, 3],
              [1, 3, -2],
              [4, -3, 1]])  # Coefficients of xi, eta, zeta

b = np.array([5, -3, 4])  # Right-hand side values

solution = np.linalg.solve(C, b)  # Solve for [xi, eta, zeta]

print("Problem 3: Solving System of Equations")
print(f"xi = {solution[0]:.2f}, eta = {solution[1]:.2f}, zeta = {solution[2]:.2f}")

#%%

C2 = np.array([[2, -1, 3, 8, 1],
              [1, 3, -2, 0, 3],
              [4, -3, 1, 4, -5],
              [6, 6, 1, 0, -9],
              [8, 9, 1, 1, 5]])  # Coefficients of xi, eta, zeta

b2 = np.array([5, -3, 4, 8, 12])  # Right-hand side values

solution2 = np.linalg.solve(C2, b2)  # Solve for [xi, eta, zeta]

print("Solving Larger System of Equations")
print(f"xi = {solution2[0]:.2f}, eta = {solution2[1]:.2f}, zeta = {solution2[2]:.2f}, theta = = {solution2[3]:.2f}, beta = = {solution2[4]:.2f}")

#%%  I prompted chatgpt with the following, 
#    "write the code needed to solve the determinate of AxB.
#    A and B are both 8x8 matrices, entered as an array of numbers"


# Define the 8x8 matrices A and B as arrays (enter your numbers in place of these arrays)
A = np.array([
    [2, -1, 3, 4, -5, 6, 7, 8],
    [1, 2, 4, 3, 6, 5, 8, 7],
    [3, 4, 2.6, 1, 0, -7, 6.9, 5],
    [4, 3, 1, 0, 7, 8, 5, 6],
    [-5, 6, 8, 7, 1, 2.5, 4, 3],
    [6, 5, 7, 8, 2, 1.1, 3, 4],
    [7, -8, 0, 6, 4, 3, 0, 2],
    [8, 7, 6, 5, 3, 4, 2, 1]
])

B = np.array([
    [0, 3, 0, 4, 5, -7, -6, 8],
    [3, -1, 4, 2, 7, -5.5, -8, 6],
    [2, 4.4, 1, 3, 6.2, 8, 5, 7],
    [4, 2, 3, 1, 8, 6, 7, 5],
    [5, 7.1, 6, 8, 1, 0, -2, 4],
    [7, 5, -8, 6, 3, 1, -4.7, 2],
    [0, 8.8, 5, 0, 2, 4, 1, 3],
    [8, 6, 7, 5, 4, 0, 3, 1]
])

# Calculate the determinant of A and B
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)

# Calculate the determinant of A * B (det(A*B) = det(A) * det(B))
det_AB = det_A * det_B

# Output the results
print("Determinant of A:", det_A)
print("Determinant of B:", det_B)
print("Determinant of A * B:", det_AB)
#%%  I prompted chatgpt with the following,
#  "create a code that solves of the inverse of a matrix C."
#  
import matplotlib.pyplot as plt


def inverse_matrix(C):
    try:
        C_inv = np.linalg.inv(C)
        return C_inv
    except np.linalg.LinAlgError:
        return "Matrix is singular and cannot be inverted."

# Example usage
C = np.array([[4, 3, 2, 1, 5, 6, 7, 8],
              [3, 4, 1, 2, 7, 8, 5, 6],
              [2, 1, 4, 3, 6, 5, 8, 7],
              [1, 2, 3, 4, 8, 7, 6, 5],
              [5, 7, 6, 8, 4, 3, 2, 1],
              [6, 5, 8, 7, 3, 4, 1, 2],
              [7, 8, 5, 6, 2, 1, 4, 3],
              [8, 6, 7, 5, 1, 2, 3, 4]])
C_inv = inverse_matrix(C)
print("Inverse of C:")
print(C_inv)
plt.matshow(C_inv)
plt.plot()
#%%  I prompted chatgpt with the following,
#    "write a code that iteratively multiplies a matrix 
#    by itself until it reaches steady values. make the code 
#    determine the amount of iterations it takes to reach steady values."

def iterative_matrix_multiplication(D, tol=1e-6, max_iter=1000):
    prev_D = np.copy(D)
    iterations = 0
    
    for _ in range(max_iter):
        D = np.dot(D, D)
        iterations += 1
        
        if np.allclose(D, prev_D, atol=tol):
            return D, iterations
        
        prev_D = np.copy(D)
    
    return D, iterations

# Example usage
D = np.array([[0.5, 0.25, 0.25,],
              [0.2, 0.7, 0.1,],
              [0.3, 0.3, 0.4,],
              ])
steady_D, num_iterations = iterative_matrix_multiplication(D)
print("Steady matrix:")
print(steady_D)
print("Iterations to reach steady values:", num_iterations)

#%%   I prompted chatgpt with the following,
#     "create a code that computes the eigenvalues 
#     and eigenvalues of the system by solving Kv=w^2*Mv"
#    I would love to explain more but I have no clue what eigenvalues and eigenvectors are

def solve_eigenproblem(K, M):
    """Solves the generalized eigenvalue problem Kv = w^2 * Mv."""
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)
    frequencies = np.sqrt(np.abs(eigenvalues))  # Taking absolute value to avoid complex results
    return frequencies, eigenvectors


# Example eigenproblem
K = np.array([[8, -5, 0, 0, 0,],
              [-5, 9, -4, 0, 0,],
              [0, -4, 10, -6, 0,],
              [0, 0, -6, 8, -2,],
              [0, 0, 0, -2, 9,],
             ])
M = np.array([[1, 0, 0, 0, 0,],
              [0, 2, 0, 0, 0,],
              [0, 0, 1, 0, 0,],
              [0, 0, 0, 2, 0,],
              [0, 0, 0, 0, 3,],])
frequencies, eigenvectors = solve_eigenproblem(K, M)
print("Eigenfrequencies:")
print(frequencies)
print("Eigenvectors:")
print(eigenvectors)

#%%

