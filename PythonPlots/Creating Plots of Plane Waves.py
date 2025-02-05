# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:00:41 2025

@author: dylan
"""

"""I used a lot of my code from the previous assignment to help guide me throuhg this assignment."""

import numpy as np
import matplotlib.pyplot as plt

"""First four Euler functions"""
def function(t):
    
    return 2*np.exp(-1j*(4*t+5))

x = np.linspace (-7, 7, 1000)
y = function(x)

plt.plot(x, y, color='purple')

def function(t):
    
    return 3*np.exp(-1j*(8*t+2))

x = np.linspace (-7, 7, 1000)
y = function(x)

plt.plot(x, y, color='green')

def function(t):

    return 1*np.exp(-1j*(33*t+17))

x = np.linspace (-7, 7, 1000)
y = function(x)

plt.plot(x, y, color='blue')

def function(t):
    
    return 9*np.exp(-1j*(1*t+765))

x = np.linspace (-7, 7, 1000)
y = function(x)

plt.plot(x, y, color='red')

#%%
