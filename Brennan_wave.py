#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:18:12 2025

@author: brennan
"""

import numpy as np 
import matplotlib.pyplot as plt 
"Below is the equations for each wave function in which I define what A, omega, and phi are for the function. The statements next to each function help to define what graph it goes to."
# Define the wave function
A1 = 5  # Amplitude for first wave function 
omega1 = 4 # Angular Frequency for first wave function 
phi1 = 7 # Phase Shift for first wave function 

A2 = 2 #amplitude for second wave function 
omega2 = 6 #angular frequency for second wave function 
phi2 = 4 #phase shift for second wave function 

A3 = 7 #amplitude for third wave function 
omega3 = 9 #angular frequency for second wave function 
phi3 = 3 #phase shift for second wave function 

A4 = 3 #amplitude for fourth wave function
omega4 = 5 #angular frequency for fourth wave function 
phi4 = 6 #phase shift for fourth wave function 
def wave_function (t, A, omega, phi):
    return A * np.exp(-1j * (omega * t + phi))
"The time range helps to define how big the function will be."
# Time Range
t = np.linspace(-2*np.pi, 2*np.pi, 400)
psi_t1 = wave_function(t, A1, omega1, phi1)
psi_t2 = wave_function(t, A2, omega2, phi2)
psi_t3 = wave_function(t, A3, omega3, phi3)
psi_t4 = wave_function(t, A4, omega4, phi3)
psi_t_combined = psi_t1 + psi_t2 + psi_t3 +psi_t4
"Below is a script that plots the real and imaginary parts of the function."
# Extract Real and Imaginary parts
real_part1 = np.real (psi_t1)
imag_part1 = np.imag (psi_t1)
real_part2 = np.real(psi_t2)
imag_part2 = np.imag(psi_t2)
real_part3 = np.real(psi_t3)
imag_part3 = np.imag(psi_t3)
real_part4 = np.real(psi_t4)
imag_part4 = np.imag(psi_t4)
real_part_combined = np.real(psi_t_combined)
imag_part_combined = np.imag(psi_t_combined)
"The 1-4 plot wave functions interact with the (Define wave function and time range) to plot each function individually."
# Plot first wave function 
plt.figure(figsize=(10, 5))
plt.plot(t, real_part1, label='Real Part of $5e^{-i(4t+7)}$', color='b')
plt.plot(t, imag_part1, label='Imaginary Part of $5e^{-i(4t+7)}$', color='r', linestyle='dashed')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('Time (t)')
plt.ylabel('Wave Function')
plt.title('Wave Function Plot for $5e^{-i(4t+7)}$')
plt.legend()
plt.grid()
plt.show()

# Plot second wave function 
plt.figure(figsize=(10,5))
plt.plot (t, real_part2, label = 'Real Part of $2e^{-i(6t+4)}$', color='b')
plt.plot(t, imag_part2, label='Imaginary Part of $2e^{-i(6t+4)}$', color='r', linestyle = 'dashed')
plt.axhline(0, color= 'black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('Time (t)')
plt.ylabel('Wave Function')
plt.title('Wave Function Plot for $2e^{-i(6t+4)}$')
plt.legend()
plt.grid()
plt.show()

# Plot third wave function 
plt.figure(figsize = (10,5))
plt.plot(t, real_part3, label= 'Real Part of $7e^{-i(9t+3)}$', color= 'b')
plt.plot(t, imag_part3, label= 'Imaginary Part of $7e^{-i(9t+3)}$', color= 'r', linestyle= 'dashed')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('Time(t)')
plt.ylabel('Wave Function')
plt.title('Wave Function Plot for $7e^{-i(9t+3)}$')
plt.legend()
plt.grid()
plt.show()

# Plot fourth wave function 
plt.figure(figsize= (10,5))
plt.plot(t, real_part4, label='Real Part of $3e^{-i(5t+6)}$', color='b')
plt.plot(t, imag_part4, label= 'Imaginary Part of $3e^{-i(5t+6)}$', color='r', linestyle= 'dashed')
plt.axhline(0, color= 'black', linewidth=0.5)
plt.axvline(0, color= 'black', linewidth=0.5)
plt.xlabel('Time (t)')
plt.ylabel('Wave Function')
plt.title('Wave Function Plot for $3e^{-i(5t+6)}$')
plt.legend()
plt.grid()
plt.show()
"The combined wave function script is the same as the previuos 4 except it just combines the 4 into one function. "
# Plot combined wave functions 
plt.figure(figsize = (10,5))
plt.plot(t, real_part_combined, label= 'Real Part of Combined Wave Functions', color = 'b')
plt.plot(t, imag_part_combined, label= 'Imaginary Part of Combined Wave Functions', color= 'r', linestyle= 'dashed')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('Time (t)')
plt.ylabel('Wave Function')
plt.title('Combined Wave Function Plot')
plt.legend()
plt.grid()
plt.show()
