#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 08:29:15 2025

@author: lochstu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0   # Length of pendulum (m)

# Define the system of first-order ODEs
def nonlinear_pendulum(t, y):
    theta, omega = y  # Unpack state variables (angular displacement and velocity)
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Define a range of initial conditions for phase space visualization
initial_conditions = [
    [0.1, 0],   # Small angle, zero velocity
    [0.5, 0],   # Moderate angle
    [1.5, 0],   # Large angle
    [3.0, 0],   # Very large angle
    [0, 1.0],   # Zero angle, initial velocity
    [0, -1.0],  # Zero angle, negative velocity
    [np.pi / 2, 0.5],  # Mixed initial conditions
    [-np.pi / 2, -0.5], # Mixed initial conditions (negative)
]

# Time span for the solution
t_span = (0, 10)  # From t=0 to t=10 seconds
t_eval = np.linspace(*t_span, 1000)  # Time points for evaluation

# Plot the phase space diagram
plt.figure(figsize=(8, 6))

for y0 in initial_conditions:
    solution = solve_ivp(nonlinear_pendulum, t_span, y0, t_eval=t_eval)
    theta, omega = solution.y
    plt.plot(theta, omega, label=f"IC: θ={y0[0]:.2f}, ω={y0[1]:.2f}")

# Labels and title
plt.xlabel("Angular Displacement (θ in radians)")
plt.ylabel("Angular Velocity (ω in rad/s)")
plt.title("Phase Space Diagram of a Nonlinear Pendulum")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.legend()
plt.show()
