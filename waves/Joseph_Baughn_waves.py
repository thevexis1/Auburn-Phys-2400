# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:28:00 2025

@author: baugh
"""
import numpy as np
import matplotlib.pyplot as plt

# Set up the time values
t = np.linspace(0, 2 * np.pi, 1000)

# Choose the values for each wave
wave_settings = [
    (1, 1, 0),               # Wave 1
    (0.8, 2, np.pi / 4),     # Wave 2
    (0.6, 3, np.pi / 2),     # Wave 3
    (1.2, 0.5, -np.pi / 3)   # Wave 4
]

# Euler's formula says: e^(iθ) = cos(θ) + i*sin(θ)
waves = []
for A, omega, phi in wave_settings:
    wave = A * np.exp(1j * (omega * t + phi))  # This creates the wave
    waves.append(wave)

# Plot all waves (real and imaginary parts) on one graph
# Real part = cosine wave 
# Imaginary part = sine wave
plt.figure(figsize=(10, 6))
for i, wave in enumerate(waves):
    plt.plot(t, wave.real, label=f'Wave {i+1} Real', linestyle='-')
    plt.plot(t, wave.imag, label=f'Wave {i+1} Imag', linestyle='--')
plt.title('Real and Imaginary Parts of Each Wave')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# Plot only the real parts in a new graph
plt.figure(figsize=(10, 5))
for i, wave in enumerate(waves):
    plt.plot(t, wave.real, label=f'Wave {i+1} Real')
plt.title('Only Real Parts')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot only the imaginary parts in a new graph
plt.figure(figsize=(10, 5))
for i, wave in enumerate(waves):
    plt.plot(t, wave.imag, label=f'Wave {i+1} Imag')
plt.title('Only Imaginary Parts')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Add all the waves together and plot the result
combined_wave = sum(waves)

plt.figure(figsize=(10, 6))
plt.plot(t, combined_wave.real, label='Combined Real Part', color='blue')
plt.plot(t, combined_wave.imag, label='Combined Imag Part', color='red', linestyle='--')
plt.title('Combined Wave')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()