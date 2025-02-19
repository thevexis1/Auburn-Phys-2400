# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:00:41 2025

@author: dylan
"""

"""I used a lot of my code from the previous assignmentg to help 
guide me throuhg this assignment. I also had help from Gemini AI
to create my code."""

import numpy as np
import matplotlib.pyplot as plt

#%%First four Euler functions

"""FUNCTION #1"""
def function(t):
    
    return 2*np.exp(-1j*(4*t+5))

t = np.linspace (-7, 7, 1000)
y = function(t)

plt.plot(t, y, color='purple') 
"""makes plot with color"""

"""FUNCTION #2"""
def function(t):
    
    return 3*np.exp(-1j*(8*t+2))

t = np.linspace (-7, 7, 1000)
y = function(t)

plt.plot(t, y, color='green') 
"""makes plot with color"""

"""FUNCTION #3"""
def function(t):

    return 1*np.exp(-1j*(33*t+17))

t = np.linspace (-7, 7, 2000)
y = function(t)

plt.plot(t, y, color='blue') 
"""makes plot with color"""

"""FUNCTION #4"""
def function(t):
    
    return 9*np.exp(-1j*(1*t+765))

t = np.linspace (-7, 7, 1000)
y = function(t)

plt.plot(t, y, color='red') 
"""makes plot with color"""

#%% Same functions broken into their real and imaginary parts

"""FUNCTION #1: 2*np.exp(-1j*(4*t+5))"""
def function(t):
    
    angle = -1*(4*t+5) 
    """the angle in the euler exponent"""
    real = 2*(np.cos(angle)) 
    """Real part of the euler"""
    imag = - 2j*(np.sin(angle)) 
    """imaginary part of the euler"""
    return real + imag 
"""equation of orignal function"""

t = np.linspace (-7, 7, 1000)
y = function(t)

real = np.real(y) 
"""extracts real part of complex number"""
imag = np.imag(y) 
"""extracts imaginary part of complex number"""

plt.plot(real, imag, color='purple') 
"""makes plot with color"""
plt.axis('equal') 
"""makes sure real and imaginary parts are plotted with the same scale"""

"""FUNCTION #2: 3*np.exp(-1j*(8*t+2))"""
def function(t):
    
    angle = -1*(8*t+2) 
    """the angle in the euler exponent"""
    real = 3*(np.cos(angle)) 
    """real part of the euler"""
    imag = - 3j*(np.sin(angle)) 
    """imaginary part of the euler"""
    return real + imag

t = np.linspace (-7, 7, 1000)
y = function(t)

real = np.real(y) 
"""extracts real part of complex number"""
imag = np.imag(y) 
"""extracts imaginary part of complex number"""

plt.plot(real, imag, color='green') 
"""makes plot with color"""
plt.axis('equal') 
"""makes sure real and imaginary parts are plotted with the same scale"""

"""FUNCTION #3: 1*np.exp(-1j*(33*t+17))"""
def function(t):
    
    angle = -1*(33*t+17)  
    """the angle in the euler exponent"""
    real = 1*(np.cos(angle)) 
    """real part of the euler"""
    imag = - 1j*(np.sin(angle)) 
    """imaginary part of the euler"""
    return real + imag

t = np.linspace (-7, 7, 1000)
y = function(t)

real = np.real(y) 
"""extracts real part of complex number"""
imag = np.imag(y) 
"""extracts imaginary part of complex number"""

plt.plot(real, imag, color='blue') 
"""makes plot with color"""
plt.axis('equal') 
"""makes sure real and imaginary parts are plotted with the same scale"""

"""FUNCTION #4: 9*np.exp(-1j*(1*t+765))"""
def function(t):
    
    angle = -1*(1*t+765) 
    """the angle in the euler exponent"""
    real = 9*(np.cos(angle)) 
    """real part of the euler"""
    imag = - 9j*(np.sin(angle)) 
    """imaginary part of the euler"""
    return real + imag

t = np.linspace (-7, 7, 1000)
y = function(t)

real = np.real(y) 
"""extracts real part of complex number"""
imag = np.imag(y) 
"""extracts imaginary part of complex number"""

plt.plot(real, imag, color='red') 
"""makes plot with color"""
plt.axis('equal') 
"""makes sure real and imaginary parts are plotted with the same scale"""

#%% Seperate plot of all four waves added together

"""COMBINED FUNCTION"""
def funtion(t):
    
    return 2*np.exp(-1j*(4*t+5)) + 3*np.exp(-1j*(8*t+2)) +  1*np.exp(-1j*(33*t+17)) + 9*np.exp(-1j*(1*t+765))
"""I simply just added all four of the functions together"""

t = np.linspace (-7, 7, 1000)
y = function(t)

plt.plot(t, y, color='purple') 
"""makes plot with color"""