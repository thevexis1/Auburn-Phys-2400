#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 23:59:26 2025

@author: lochstu
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_potential(L, Nx, Ny, V0, n=5, tol=1e-4, omega=1.5):
    dx = 1 / Nx  # Step in x-direction
    dy = L / Ny  # Step in y-direction
    
    V = np.zeros((Nx+1, Ny+1))  # Initialize potential grid
    V[0, :] = V0 * np.sin(n * np.pi * np.linspace(0, L, Ny+1) / L)  # Sinusoidal potential at x=0
    
    error = 1
    while error > tol:
        V_old = V.copy()
        for i in range(1, Nx):  # Skip x=0 boundary
            for j in range(1, Ny):  # Skip y=0 and y=L boundaries
                V[i, j] = (1 - omega) * V_old[i, j] + omega * 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
        
        V[-1, :] = V[-2, :]  # Ensure decay as x -> infinity
        
        error = np.max(np.abs(V - V_old))
    
    return V

# Define parameters
L = 1.0  # Plate height
Nx = 50  # Grid points in x-direction
Ny = 50  # Grid points in y-direction
V0 = 1.0  # Maximum potential at x=0
n = 5  # Number of oscillations in y-direction

V = solve_potential(L, Nx, Ny, V0, n)


# Plot the result
X, Y = np.meshgrid(np.linspace(0, 1, Nx+1), np.linspace(0, L, Ny+1))
plt.contourf(X, Y, V.T, levels=50, cmap='inferno')
plt.colorbar(label='Potential V(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Electrical Potential Distribution with Oscillations')
plt.show()


# Surface plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, V.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Potential V(x,y)')
ax.set_title('3D Surface Plot of Electrical Potential')
plt.show()