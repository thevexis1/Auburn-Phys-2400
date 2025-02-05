         `# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:18:47 2025

@author: dylan
"""

import numpy as np
import matplotlib.pyplot as plt

i=5+8j
print(i)
k=62-7j
m=i+k
print (round(m.real))
print (round(m.imag))
r, theta = abs(m), np.angle(m)
print (r, theta)
exp = r*np.exp(theta*1j)
print (exp)
print (theta)
x = np.linspace(-10, 10, 400)
xexp = r*np.exp(x*1j)
plt.figure(figsize=(6, 6))
plt.plot(xexp)



plt.xlabel("Re")
plt.ylabel("Im")

plt.show()