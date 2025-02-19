#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:09:51 2025

@author: lochstu

Here is the ChapGPT prompt I used the following to make a square wave
'write a python code to plot a square wave between x= 0 and 6, with one peak and one trough'

and the following to make the fourier code:
    'write a python code to calculate a Fourier series for a square wave'

"""

import numpy as np
import matplotlib.pyplot as plt

def fourier_series_square_wave(t, terms=10):
    """Compute the Fourier series approximation of a square wave."""
    result = np.zeros_like(t)
    for n in range(1, terms * 2, 2):  # Only odd harmonics
        result += (4 / (np.pi * n)) * np.sin(n * t)
    return result

# Time variable
t = np.linspace(0, 2 * np.pi, 2000)


# Define square wave function
square_wave = np.where(t < 3, 1, -1)  # Peak for x < 3, trough for x >= 3


# Compute Fourier series for square wave
square_wave10 = fourier_series_square_wave(t, terms=10)
square_wave100 = fourier_series_square_wave(t, terms=100)
square_wave1000 = fourier_series_square_wave(t, terms=1000)

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(t, square_wave10, 'r', label='Fourier series using 10 terms')
plt.plot(t, square_wave100, 'b', label='Fourier series using 100 terms')
plt.plot(t, square_wave1000, 'k', label='Fourier series using 1000 terms')
plt.plot(t, square_wave, label='Square Wave')
plt.title("Fourier Series Approximation of a Square Wave")
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()
