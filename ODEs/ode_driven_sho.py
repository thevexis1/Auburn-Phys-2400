#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 08:20:44 2025

@author: lochstu

ChatGPT prompt: "write a python code to solve a driver harmonic oscillator ode"
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
omega_0 = 2.0      # Natural frequency (rad/s)
beta = 0.2         # Damping coefficient
F_0 = 1.0          # Driving force amplitude
omega_drive = 1.5  # Driving frequency

# Define the system of first-order ODEs
def driven_harmonic_oscillator(t, y):
    x, v = y  # Unpack state variables (position and velocity)
    dxdt = v
    dvdt = -2 * beta * v - omega_0**2 * x + F_0 * np.cos(omega_drive * t)
    return [dxdt, dvdt]

# Initial conditions: x(0) = 0, v(0) = 0
y0 = [0.0, 0.0]

# Time span for the solution
t_span = (0, 100)  # From t=0 to t=50
t_eval = np.linspace(*t_span, 1000)  # Time points for evaluation

# Solve the ODE
solution = solve_ivp(driven_harmonic_oscillator, t_span, y0, t_eval=t_eval)

# Extract results
t = solution.t
x = solution.y[0]  # Position
v = solution.y[1]  # Velocity

# Plot the displacement vs. time
plt.figure(figsize=(8, 5))
plt.plot(t, x, label="Displacement (x)", color='b')
plt.xlabel("Time (s)")
plt.ylabel("Displacement (x)")
plt.title("Driven Harmonic Oscillator - Displacement vs. Time")
plt.legend()
plt.grid()
plt.show()

# Plot the phase space diagram (x vs. v)
plt.figure(figsize=(8, 6))
plt.plot(x, v, color='r')
plt.xlabel("Displacement (x)")
plt.ylabel("Velocity (v)")
plt.title("Phase Space Diagram of a Driven Harmonic Oscillator")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.show()