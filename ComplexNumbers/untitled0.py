#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:17:48 2025

@author: auttieeehill
"""
import numpy as np
import matplotlib.pyplot as plt
#%% Part 1: Algebra of Complex Numbers
z=2+4j
print(z)
y=4.3-9.2j
x=z+y
print (round(x.real))
print (round(x.imag))

r, theta = abs(x), np.angle(x)
print (r, theta)
x_exp = r * np.exp(1j * theta)
print("Exponential form of z:", x_exp)
#%% Part 4: Plotting in the Complex Plane
xaxis = np.linspace(-10, 10, 400)
xexp = r * np.exp(1j * xaxis)
plt.figure(figsize=(6, 6))
plt.plot(xexp)


plt.xlabel("Re")
plt.ylabel("Im")
plt.show()