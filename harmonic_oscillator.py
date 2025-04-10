#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:54:42 2025

@author: lochstu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
omega = 2.0  # Natural frequency (rad/s)

# Define the system of first-order ODEs
def harmonic_oscillator(t, y):
    x, v = y  # Unpack state variables (position and velocity)
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

# Initial conditions: x(0) = 1, v(0) = 0
y0 = [1.0, 0.0]

# Time span for the solution
t_span = (0, 10)  # From t=0 to t=10
t_eval = np.linspace(*t_span, 100)  # Time points for evaluation

# Solve the ODE
solution = solve_ivp(harmonic_oscillator, t_span, y0, t_eval=t_eval)

# Extract results
t = solution.t
x = solution.y[0]  # Position

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(t, x, label="Displacement (x)", color='b')
plt.xlabel("Time (s)")
plt.ylabel("Displacement (x)")
plt.title("Simple Harmonic Oscillator")
plt.legend()
plt.grid()
plt.show()