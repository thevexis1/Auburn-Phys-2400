#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:06:33 2025

@author: lochstu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the function to be approximated
def user_function(x):
    return np.exp(-x**2)*np.sin(6.*x)  # User can modify this function

# Fourier series coefficients
def fourier_coefficients(xmin,xmax, f, n_terms=10):
    xrange=xmax-xmin
    a0 =  (2./xrange)*quad(lambda x: f(x), xmin, xmax)[0]
    an = [(2./xrange)*quad(lambda x: f(x) * np.cos(2.*n * np.pi* x/xrange), xmin, xmax)[0] for n in range(1, n_terms + 1)]
    bn = [(2./xrange)*quad(lambda x: f(x) * np.sin(2.*n * np.pi* x/xrange), xmin, xmax)[0] for n in range(1, n_terms + 1)]
    return a0, an, bn

# Fourier series reconstruction
def fourier_series(x, a0, an, bn, n_terms=10):
    result = a0 / 2
    xrange=x[-1]-x[0]
    for n in range(1, n_terms + 1):
        result += an[n-1] * np.cos(2.*n * np.pi* x/xrange) + bn[n-1] * np.sin(2.*n * np.pi* x/xrange)
    return result

# Parameters
xmin=-6.0
xmax=6.0  # Period
n_terms = 20  # Number of terms in the Fourier series
x = np.linspace(xmin,xmax, 1000)

# Compute coefficients
a0, an, bn = fourier_coefficients(xmin,xmax,user_function, n_terms)

# Compute Fourier approximation
y_approx = fourier_series(x, a0, an, bn, n_terms)

# Plot original function and Fourier series approximation
plt.plot(x, user_function(x), label='Original Function')
plt.plot(x, y_approx, label=f'Fourier Series ({n_terms} terms)', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Fourier Series Approximation')
plt.grid()
plt.show()
