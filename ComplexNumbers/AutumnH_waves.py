#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:14:27 2025

@author: auttieeehill
"""
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the wave functions
A1, A2, A3, A4 = 1, 2, 0.5, 1.5  # Amplitudes
omega1, omega2, omega3, omega4 = 2, 3, 1, 2.5  # Angular frequencies (ω)
phi1, phi2, phi3, phi4 = 0, np.pi/4, np.pi/2, -np.pi/4  # Phases (φ)

# Time vector (t)
t = np.linspace(0, 10, 1000)

# Define the four wave functions
def wave(A, omega, phi, t):
    return A * np.exp(-1j * (omega * t + phi))

# Create the individual wave functions
wave1 = wave(A1, omega1, phi1, t)
wave2 = wave(A2, omega2, phi2, t)
wave3 = wave(A3, omega3, phi3, t)
wave4 = wave(A4, omega4, phi4, t)

# Real and Imaginary parts
real_wave1, imag_wave1 = np.real(wave1), np.imag(wave1)
real_wave2, imag_wave2 = np.real(wave2), np.imag(wave2)
real_wave3, imag_wave3 = np.real(wave3), np.imag(wave3)
real_wave4, imag_wave4 = np.real(wave4), np.imag(wave4)

# Plot the real parts
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(t, real_wave1, label="Wave 1 (Real)", color='pink')
plt.plot(t, real_wave2, label="Wave 2 (Real)", color='mediumslateblue')
plt.plot(t, real_wave3, label="Wave 3 (Real)", color='cornflowerblue')
plt.plot(t, real_wave4, label="Wave 4 (Real)", color='darkseagreen')
plt.title("Real Parts of Waves")
plt.xlabel("Time (t)")
plt.ylabel("Real Part")
plt.legend()

# Plot the imaginary parts
plt.subplot(2, 2, 2)
plt.plot(t, imag_wave1, label="Wave 1 (Imag)", color='slategrey')
plt.plot(t, imag_wave2, label="Wave 2 (Imag)", color='thistle')
plt.plot(t, imag_wave3, label="Wave 3 (Imag)", color='cornflowerblue')
plt.plot(t, imag_wave4, label="Wave 4 (Imag)", color='darkseagreen')
plt.title("Imaginary Parts of Waves")
plt.xlabel("Time (t)")
plt.ylabel("Imaginary Part")
plt.legend()

# Combine the waves (sum of all four)
combined_wave = wave1 + wave2 + wave3 + wave4
real_combined = np.real(combined_wave)
imag_combined = np.imag(combined_wave)

# Plot the combined wave (Real part)
plt.subplot(2, 2, 3)
plt.plot(t, real_combined, label="Combined Wave (Real)", color='orchid')
plt.title("Combined Wave - Real Part")
plt.xlabel("Time (t)")
plt.ylabel("Real Part")

# Plot the combined wave (Imaginary part)
plt.subplot(2, 2, 4)
plt.plot(t, imag_combined, label="Combined Wave (Imag)", color='lightsteelblue')
plt.title("Combined Wave - Imaginary Part")
plt.xlabel("Time (t)")
plt.ylabel("Imaginary Part")

# Show the plots
plt.tight_layout()
plt.show()

