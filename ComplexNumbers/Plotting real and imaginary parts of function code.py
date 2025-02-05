#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:44:53 2025

@author: lillikonrad
"""
import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 10, 1000)

# Parameters for the four wave functions
A1, omega1, phi1 = 1, 2, 0
A2, omega2, phi2 = 1.5, 3, np.pi/4
A3, omega3, phi3 = 2, 1, np.pi/2
A4, omega4, phi4 = 0.5, 4, -np.pi/6

# Wave functions
f1 = A1 * np.exp(-1j * (omega1 * t + phi1))
f2 = A2 * np.exp(-1j * (omega2 * t + phi2))
f3 = A3 * np.exp(-1j * (omega3 * t + phi3))
f4 = A4 * np.exp(-1j * (omega4 * t + phi4))

# Real and Imaginary parts
f1_real = np.real(f1)
f1_imag = np.imag(f1)

f2_real = np.real(f2)
f2_imag = np.imag(f2)

f3_real = np.real(f3)
f3_imag = np.imag(f3)

f4_real = np.real(f4)
f4_imag = np.imag(f4)

# Combined wave function
f_combined = f1 + f2 + f3 + f4
f_combined_real = np.real(f_combined)
f_combined_imag = np.imag(f_combined)

# Plotting the individual wave functions in one figure

plt.figure(figsize=(14, 10))

# Subplot for Wave 1 (Real & Imag)
plt.subplot(4, 2, 1)
plt.plot(t, f1_real, label="Real Part", color='blue')
plt.plot(t, f1_imag, label="Imaginary Part", color='red', linestyle='--')
plt.title('Wave 1')
plt.legend()

# Subplot for Wave 2 (Real & Imag)
plt.subplot(4, 2, 3)
plt.plot(t, f2_real, label="Real Part", color='blue')
plt.plot(t, f2_imag, label="Imaginary Part", color='red', linestyle='--')
plt.title('Wave 2')
plt.legend()

# Subplot for Wave 3 (Real & Imag)
plt.subplot(4, 2, 5)
plt.plot(t, f3_real, label="Real Part", color='blue')
plt.plot(t, f3_imag, label="Imaginary Part", color='red', linestyle='--')
plt.title('Wave 3')
plt.legend()

# Subplot for Wave 4 (Real & Imag)
plt.subplot(4, 2, 7)
plt.plot(t, f4_real, label="Real Part", color='blue')
plt.plot(t, f4_imag, label="Imaginary Part", color='red', linestyle='--')
plt.title('Wave 4')
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# Plotting the combined wave functions in a separate figure

plt.figure(figsize=(12, 6))

# Subplot for Combined Wave (Real Part)
plt.subplot(2, 1, 1)
plt.plot(t, f_combined_real, label="Real Part", color='blue')
plt.title('Combined Wave (Real Part)')
plt.legend()

# Subplot for Combined Wave (Imaginary Part)
plt.subplot(2, 1, 2)
plt.plot(t, f_combined_imag, label="Imaginary Part", color='red')
plt.title('Combined Wave (Imaginary Part)')
plt.legend()

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


# Adjust layout for better spacing
plt.tight_layout()
plt.show()
