#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:38:48 2025

@author: AlexDickerson
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
a=3
b=2
c=2
def y1(x):
    return (a*np.exp(-((x/b)**2)))*((np.cos(c*x))**2)
    
def y2(x):
    return (a*np.exp(-((x/b)**2)))


x=np.linspace(-7,7,500)
plt.plot(x,y1(x),color='teal', alpha=.5)
plt.plot(x,y2(x),color='red', alpha=.5)

plt.title("Gaussian Envelope")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

    
plt.show




#%%