# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 07:54:30 2025

@author: dylan
"""

"""I used Gemini AI and previous code in class to help guide me throuhg this."""

import numpy as np
import matplotlib.pyplot as plt
    
"""First function: the one that I varied (purple)."""

def function(x):
    

    return 5*np.exp(-(x/2)**2) * np.cos(3*x)**2
   
x = np.linspace (-7, 7, 1000)
y = function(x)

plt.plot(x, y, color='purple')

"""Second function: the one for comparison (blue and dotted)."""

def function(r):
    
    return np.exp(-(r)**2) * np.cos(r)**2

r = np.linspace (-7, 7, 1000)
y = function(x)

plt.plot(x, y, color='green', linestyle='dotted')