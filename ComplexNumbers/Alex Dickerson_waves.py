#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:31:26 2025

@author: johnalexander
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
A1=5
Ω1=.75
phi1=0

j=1j


def f1(t):
    return A1*(np.exp(-j*((Ω1*t)+phi1)))



A2=4
Ω2=1.5
phi2=0


def f2(t):
    return A2*(np.exp(-j*((Ω2*t)+phi2)))

A3=3
Ω3=2.5
phi3=0

def f3(t):
    return A3*(np.exp(-j*((Ω3*t)+phi3)))

A4=2
Ω4=4
phi4=0


def f4(t):
    return A4*(np.exp(-j*((Ω4*t)+phi4)))



t=np.linspace(-5,5,500)


plt.plot(t,f1(t),color='red', alpha=.4)
plt.plot(t,f2(t),color='yellow', alpha=.6)
plt.plot(t,f3(t),color='teal', alpha=.8)
plt.plot(t,f4(t), color='violet', alpha=1)

plt.title("Four Waves")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()
#%%


#%%
"Graph of Real & Imaginary parts of Wave 1 with real and imaginary functions"
def real_f1(t):
    return np.real(f1(t))
def imag_f1(t):
    return np.imag(f1(t))

t=np.linspace(-5,5,500)

plt.figure(figsize=(6.4,4.8))
plt.plot(t,real_f1(t), color='red', alpha=.5, label="Real part")
plt.plot(t,imag_f1(t), color='blue', alpha=.5, label='Imaginary Part')

plt.title("Complex Components of Wave 1")
plt.xlabel('time')
plt.ylabel('Real and Imaginary Parts')

plt.show()
#%%
"Graph of Real & Imaginary parts of Wave 2 with real and imaginary functions"
def real_f2(t):
    return np.real(f2(t))
def imag_f2(t):
    return np.imag(f2(t))

t=np.linspace(-5,5,500)

plt.figure(figsize=(6.4,4.8))
plt.plot(t,real_f2(t), color='red', alpha=.5, label="Real part")
plt.plot(t,imag_f2(t), color='blue', alpha=.5, label='Imaginary Part')

plt.title("Complex Components of Wave 2")
plt.xlabel('time')
plt.ylabel('Real and Imaginary Parts')

plt.show()
#%%
"Graph of Real & Imaginary parts of Wave 3 with real and imaginary functions"
def real_f3(t):
    return np.real(f3(t))
def imag_f3(t):
    return np.imag(f3(t))

t=np.linspace(-5,5,500)

plt.figure(figsize=(6.4,4.8))
plt.plot(t,real_f3(t), color='red', alpha=.5, label="Real part")
plt.plot(t,imag_f3(t), color='blue', alpha=.5, label='Imaginary Part')

plt.title("Complex Components of Wave 3")
plt.xlabel("time")
plt.ylabel('Real and Imaginary Parts')

plt.show()
#%%
"Graph of Real & Imaginary parts of Wave 4 with real and imaginary functions"
def real_f4(t):
    return np.real(f4(t))
def imag_f4(t):
    return np.imag(f4(t))

t=np.linspace(-5,5,500)


plt.figure(figsize=(6.4,4.8))
plt.plot(t,real_f4(t), color='red', alpha=.5, label="Real part")
plt.plot(t,imag_f4(t), color='blue', alpha=.5, label='Imaginary Part')

plt.title("Complex Components of Wave 4")
plt.xlabel("time")
plt.ylabel('Real and Imaginary Parts')

plt.show()
#%%
"This cell contains a function which adds together the four wave functions, f1, f2, f3, f4. It also plots the result"

def added_waves(t):
    return f1(t)+f2(t)+f3(t)+f4(t)

plt.figure(figsize=(6.4,4.8))
plt.plot(t,added_waves(t), color='Purple', alpha=.5, label="Combined Waves")

plt.title("Combined Waves")
plt.xlabel("time")
plt.ylabel('Wave Function')

plt.show()
#%%