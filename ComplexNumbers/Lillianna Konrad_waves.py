#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:06:39 2025

@author: lillikonrad
"""

import numpy as np
import matplotlib.pyplot as plt

# Time array- goes from 0 to 10 with 1000 intervals 
t = np.linspace(0, 10, 1000)

# Parameters for the four wave functions- this defines the terms of the functions,which are
# A (multiplication value), omega and phi
A1, omega1, phi1 = 1, 4, 0
A2, omega2, phi2 = 4, 1, np.pi/2
A3, omega3, phi3 = 2, 3, np.pi/4
A4, omega4, phi4 = 8, 2, np.pi/3

# Wave functions- defines the four different functions that will be used in the plots
f1 = A1 * np.exp(-1j * (omega1 * t + phi1))
f2 = A2 * np.exp(-1j * (omega2 * t + phi2))
f3 = A3 * np.exp(-1j * (omega3 * t + phi3))
f4 = A4 * np.exp(-1j * (omega4 * t + phi4))

# Combined wave function
f_combined = f1 + f2 + f3 + f4

# Plotting the real and imaginary parts of all 4 wave functions
plt.figure(figsize=(14, 12))

# Plot for Wave 1- subplots are used to compare all the four functions next to each other, 
# colors are defined to show difference between real and imaginary parts
plt.subplot(4, 1, 1)
plt.plot(t, np.real(f1), label="Real Part", color='blue')
plt.plot(t, np.imag(f1), label="Imaginary Part", color='red', linestyle='--')
plt.title('Wave 1')
plt.legend()

# Plot for Wave 2
plt.subplot(4, 1, 2)
plt.plot(t, np.real(f2), label="Real Part", color='blue')
plt.plot(t, np.imag(f2), label="Imaginary Part", color='red', linestyle='--')
plt.title('Wave 2')
plt.legend()

# Plot for Wave 3
plt.subplot(4, 1, 3)
plt.plot(t, np.real(f3), label="Real Part", color='blue')
plt.plot(t, np.imag(f3), label="Imaginary Part", color='red', linestyle='--')
plt.title('Wave 3')
plt.legend()

# Plot for Wave 4
plt.subplot(4, 1, 4)
plt.plot(t, np.real(f4), label="Real Part", color='blue')
plt.plot(t, np.imag(f4), label="Imaginary Part", color='red', linestyle='--')
plt.title('Wave 4')
plt.legend()

plt.tight_layout()
plt.show()

# Plotting all the wave functions together (real and imaginary parts combined),
# real and imaginary parts are not separated to see the 4 functions next to each other

plt.figure(figsize=(14, 6))

# Plot each wave function (real and imaginary combined)
plt.plot(t, np.real(f1) + np.imag(f1), label="Wave 1", color='blue', alpha=0.7)
plt.plot(t, np.real(f2) + np.imag(f2), label="Wave 2", color='green', alpha=0.7)
plt.plot(t, np.real(f3) + np.imag(f3), label="Wave 3", color='orange', alpha=0.7)
plt.plot(t, np.real(f4) + np.imag(f4), label="Wave 4", color='red', alpha=0.7)

# Plot the combined wave function (real and imaginary combined)
plt.plot(t, np.real(f_combined) + np.imag(f_combined), label="Combined Wave", color='purple', linestyle='--', linewidth=2)

# Title and labels
plt.title('All Wave Functions and Combined Wave (Real + Imaginary)')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# Plotting the real and imaginary parts of the combined wave function
# Combined wave function takes the sum of the real parts and sum of imaginary parts 
plt.figure(figsize=(12, 6))

# Combined Real Part
# There are 
plt.subplot(2, 1, 1)
plt.plot(t, np.real(f_combined), label="Real Part of Combined Wave", color='blue')
plt.title('Combined Wave (Real Part)')
plt.legend()

# Combined Imaginary Part
plt.subplot(2, 1, 2)
plt.plot(t, np.imag(f_combined), label="Imaginary Part of Combined Wave", color='red')
plt.title('Combined Wave (Imaginary Part)')
plt.legend()

plt.tight_layout()
plt.show()
