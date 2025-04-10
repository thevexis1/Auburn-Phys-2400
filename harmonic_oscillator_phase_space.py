#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:58:47 2025

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

# Initial conditions for different trajectories
initial_conditions = [
    [1.0, 0.0],  # Starts at x=1, v=0
    [0.0, 1.0],  # Starts at x=0, v=1
    [1.0, 1.0],  # Starts at x=1, v=1
    [-1.0, 0.5] # Starts at x=-1, v=0.5
]

# Time span for the solution
t_span = (0, 10)  # From t=0 to t=10
t_eval = np.linspace(*t_span, 300)  # Time points for evaluation

# Plot phase space diagram
plt.figure(figsize=(8, 6))

for y0 in initial_conditions:
    solution = solve_ivp(harmonic_oscillator, t_span, y0, t_eval=t_eval)
    x, v = solution.y
    plt.plot(x, v, label=f"IC: x={y0[0]}, v={y0[1]}")

# Labels and title
plt.xlabel("Displacement (x)")
plt.ylabel("Velocity (v)")
plt.title("Phase Space Diagram of Harmonic Oscillator")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.legend()
plt.show()