#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 07:18:57 2025

@author: bradleyandrew
"""
import numpy as np
import matplotlib.pyplot as plt

# Define time range
t = np.linspace(0, 1, 5000)  # Time from 0 to 1 second, 5000 points for higher resolution

# Define frequencies, amplitudes, and phase shifts for individual waves
frequencies = [5, 15, 30, 60]  # Example frequencies in Hz
amplitudes = [1, 0.8, 0.6, 0.4]  # Example amplitudes
phase_shifts = [0, np.pi/4, np.pi/2, np.pi]  # Phase shifts in radians

# Create individual waves with phase shifts
waves = []
for i, (freq, amp, phase) in enumerate(zip(frequencies, amplitudes, phase_shifts)):
    wave = amp * np.exp(1j * (2 * np.pi * freq * t + phase))  # Complex exponential with phase
    waves.append(wave)

# Combine waves to simulate a complex FM signal
combined_wave = np.sum(waves, axis=0) * np.exp(1j * (2 * np.pi * 2 * t))  # Base FM-like behavior

# Plot individual waves (real parts)
plt.figure(figsize=(10, 6))
for i, wave in enumerate(waves):
    plt.plot(t, wave.real, label=f"Wave {i+1}: {frequencies[i]} Hz, Phase {phase_shifts[i]:.2f} rad", alpha=0.8)
plt.title("Individual Waves (Real Part)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Plot combined FM signal (real part)
plt.figure(figsize=(10, 6))
plt.plot(t, combined_wave.real, color='black', label="Combined Signal")
plt.title("Combined Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

