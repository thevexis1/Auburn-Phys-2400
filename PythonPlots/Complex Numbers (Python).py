# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:01:52 2025

@author: dylan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:20:06 2025

@author: bradleyandrew
"""

import numpy as np
import matplotlib.pyplot as plt


#%% Part 1: Algebra of Complex Numbers
print("--- Algebra of Complex Numbers ---")
# Define complex numbers
z1 = 3 + 4j
z2 = 1 - 2j

# Addition, Subtraction, Multiplication, Division
print("z1 + z2 =", z1 + z2)
print("z1 - z2 =", z1 - z2)
print("z1 * z2 =", z1 * z2)
print("z1 / z2 =", z1 / z2)

# Conjugate and Absolute Value
print("Conjugate of z1:", np.conj(z1))
print("|z1| =", abs(z1))

#%% Part 2: Representations of Complex Numbers
print("\n--- Representations of Complex Numbers ---")
# Rectangular Form: a + bi
print("Rectangular form of z1: (a, b) =", (z1.real, z1.imag))

# Polar Form: r * e^(i*theta)
r, theta = abs(z1), np.angle(z1)
print(f"Polar form of z1: r = {r}, theta = {theta} radians")

# Exponential Form
z_exp = r * np.exp(1j * theta)
print("Exponential form of z1:", z_exp)

#%% Part 3: Solving Equations with Complex Roots
print("\n--- Solving Equations with Complex Roots ---")
from numpy.polynomial.polynomial import Polynomial

# Example 1: Solving a differential equation with complex roots
# Equation: x^2 + 4x + 5 = 0
coefficients = [1, 4, 5]  # Coefficients of the polynomial
p = Polynomial(coefficients)
roots = p.roots()
print("Roots of x^2 + 4x + 5 = 0 using Polynomial:", roots)

#%% Part 4: Plotting in the Complex Plane
print("\n--- Plotting in the Complex Plane ---")

def scatter_plot_complex_points(points):
    """Scatter plot of complex numbers in the complex plane.
    Each complex number is represented as a point (Re, Im)."""
    plt.figure(figsize=(6, 6))
    plt.scatter([p.real for p in points], [p.imag for p in points], color='blue', label='Complex Points')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title("Scatter Plot in the Complex Plane")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.legend()
    plt.show()

scatter_plot_complex_points([z1, z2, roots[0], roots[1]])

#%% Contour Plot of |z| in the Complex Plane
def contour_plot_abs():
    """Contour plot of the magnitude of a complex function (|Z|).
    Contour lines represent constant values of |Z|, showing how the magnitude varies over the complex plane."""
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    plt.figure(figsize=(6, 6))
    plt.contour(X, Y, np.abs(Z), levels=10, cmap='viridis')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title("Contour Plot of |z|")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.colorbar(label='|z|')
    plt.show()

contour_plot_abs()

#%% Color Plot of Argument (angle) in the Complex Plane
def color_plot_arg():
    """Color plot of the argument (angle) of a complex function (arg(Z)).
    Colors represent the angle of Z, showing the phase information of the complex plane."""
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - 3j * Y

    plt.figure(figsize=(6, 6))
    plt.pcolormesh(X, Y, np.angle(Z), shading='auto', cmap='hsv')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title("Color Plot of arg(z)")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.colorbar(label='arg(z) (radians)')
    plt.show()

color_plot_arg()



#%% Plotting the Gaussian Function in the Complex Plane
def plot_gaussian_function():
    """Filled contour plot of the Gaussian function exp(-|Z|^2).
    Shows the spread of the Gaussian function across the complex plane."""
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    W = np.exp(-np.abs(Z)**2)

    plt.figure(figsize=(6, 6))
    plt.contourf(X, Y, np.abs(W), levels=50, cmap='plasma')
    plt.colorbar(label='|Gaussian(z)|')
    plt.title("Gaussian Function Magnitude")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.show()

plot_gaussian_function()

#%% Plotting the Gamma Function in the Complex Plane

from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma

# Define the grid
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
W = np.abs(gamma(Z))  # Magnitude of the Gamma function

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, W, cmap='hsv', edgecolor='none')  # HSV colormap for phase representation
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='|Gamma(Z)|')

# Set labels and title
ax.set_xlabel("Re(Z)")
ax.set_ylabel("Im(Z)")
ax.set_zlabel("|Gamma(Z)|")
ax.set_title("3D Plot of the Gamma Function |Gamma(Z)|")

# Show the plot
plt.show()

