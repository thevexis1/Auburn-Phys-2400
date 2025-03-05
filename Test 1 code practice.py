#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 09:16:13 2025

@author: brennan
"""

import numpy as np 
import math
import sympy
from fractions import Fraction
import cmath
#%%
z1 = 3 + 4j
z2 = 2 - 1j
print('z1/z2 = ', z1/z2)
#%%
z = -3 + 3j
a = -3
b = 3 
#Modulus
z = math.sqrt(a**2 + b**2)
#Expressing the answer in desired form 
simplified_z = sympy.sqrt(a**2 + b**2)
print(sympy.sqrt(a**2 + b**2))

#Argument
def argument_of_z(a, b):
    arg_rad = math.atan2(b, a)  # Compute the argument in radians
    fraction = Fraction(arg_rad / math.pi).limit_denominator()  # Express in terms of pi
    return f"{fraction.numerator}π/{fraction.denominator}" if fraction.denominator != 1 else f"{fraction.numerator}π"

z_real = -3
z_imag = 3
print("Argument of z:", argument_of_z(z_real, z_imag))
#%%
z = 1 - cmath.sqrt(3) * 1j

#Compute modulus
r = abs(z)

#Compute argument (angle)
theta = cmath.phase(z)

# Express in polar form using trigonometric representation
theta_deg = theta / math.pi  # Convert to fraction of pi
polar_form = f"z = {r} * (cos({theta_deg}π) + i*sin({theta_deg}π))"

print("Modulus (r):", r)
print("Argument (theta):", theta)
print("Polar form:", polar_form)
#%%
A = np.array ([[2,3],
               [1,4]])
B = np.array([[5,1],
              [2,3]])