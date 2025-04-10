#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 23:36:00 2025

@author: lochstu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0   # Length of pendulum (m)

# Define the system of first-order ODEs
def pendulum(t, y):
    theta, omega = y  # Unpack state variables
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Initial conditions for different trajectories
initial_conditions = [
    [0.5, 0],   # Small displacement
    [1.0, 0],   # Moderate displacement
    [2.0, 0],   # Large displacement
    [3.0, 0],   # Very large displacement
    [0.0, 1.0], # Pure velocity initial condition
]

# Time span for the solution
t_span = (0, 10)  # From t=0 to t=10
t_eval = np.linspace(*t_span, 300)  # Time points for evaluation

# Plot phase space diagram
plt.figure(figsize=(8, 6))

for y0 in initial_conditions:
    solution = solve_ivp(pendulum, t_span, y0, t_eval=t_eval)
    theta, omega = solution.y
    plt.plot(theta, omega, label=f"IC: θ={y0[0]}, ω={y0[1]}")

# Labels and title
plt.xlabel("Angular Displacement (θ in radians)")
plt.ylabel("Angular Velocity (ω in rad/s)")
plt.title("Phase Space Diagram of a Simple Pendulum")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.legend()
plt.show()