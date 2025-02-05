#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:30:39 2025

@author: lillikonrad
"""

import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 10, 1000)

# Parameters for the four wave functions
A1, omega1, phi1 = 1, 2, 0
A2, omega2, phi2 = 4, 3, np.pi/4
A3, omega3, phi3 = 2, 1, np.pi/2
A4, omega4, phi4 = 6, 4, np.pi/6

# Wave functions
f1 = A1 * np.exp(-1j * (omega1 * t + phi1))
f2 = A2 * np.exp(-1j * (omega2 * t + phi2))
f3 = A3 * np.exp(-1j * (omega3 * t + phi3))
f4 = A4 * np.exp(-1j * (omega4 * t + phi4))

# Combined wave function
f_combined = f1 + f2 + f3 + f4

# Plotting all wave functions together on the same plot

plt.figure(figsize=(14, 6))

# Plot each wave function (both real and imaginary parts combined)
plt.plot(t, np.real(f1), label="Wave 1", color='blue', alpha=0.7)
plt.plot(t, np.real(f2), label="Wave 2", color='green', alpha=0.7)
plt.plot(t, np.real(f3), label="Wave 3", color='orange', alpha=0.7)
plt.plot(t, np.real(f4), label="Wave 4", color='red', alpha=0.7)

# Plot the combined wave function (both real and imaginary parts combined)
plt.plot(t, np.real(f_combined), label="Combined Wave", color='purple', linestyle='--', linewidth=2)

# Title and labels
plt.title('Overlay of All Wave Functions')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
