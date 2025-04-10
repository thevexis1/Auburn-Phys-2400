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
from sympy import Matrix
import sympy as sp
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

#Compute AB and BA
AB = np.dot(A,B)
BA = np.dot(B,A)

# Print results
print("AB:")
print(AB)
print("\nBA:")
print(BA)
#%%
A = np.array([[3,2,-1],
              [4,1,0],
              [2,5,3]])

#Compute determinate
det_A = np.linalg.det(A)

print(det_A)
#%%
A = Matrix([[2,3],
            [1,4]])

#Compute inverse 
A_inv = A.inv()

print("Inverse of the matrix in fraction form:")
print(A_inv)
#%%
def simplify_complex_fraction():
    z = (2 + 3*sp.I) / (1 - sp.I)  # Define the complex fraction using sympy
    cartesian_form = sp.simplify(z)  # Simplify the expression
    return cartesian_form

result = simplify_complex_fraction()
real_part = sp.re(result)
imag_part = sp.im(result)
print(f"Cartesian form: {real_part} + {imag_part}i")
#%%
z = -2-2j
a = -2
b = -2
#Modulus
z = math.sqrt(a**2 + b**2)
simplified_z = sympy.sqrt(a**2 + b**2)
print(sympy.sqrt(a**2 + b**2))

#Argument
def argument_of_z(a, b):
    arg_rad = math.atan2(b, a)  # Compute the argument in radians
    fraction = Fraction(arg_rad / math.pi).limit_denominator()  # Express in terms of pi
    return f"{fraction.numerator}π/{fraction.denominator}" if fraction.denominator != 1 else f"{fraction.numerator}π"

real = -2
imag = -2 
print("Argument of z:", argument_of_z(z_real, z_imag))
#%%
def solve_quadratics(a,b,c):
    discriminant = cmath.sqrt(b**2 - 4*a*c)
    root1 = (-b + discriminant)/(2*a)
    root2 = (-b - discriminant)/(2*a)
    return root1, root2 
# Given equation: z^2 - 2z + 5 = 0
a,b,c = 1,-2,5
solution = solve_quadratics(a,b,c)
print("Solutions for z:", solution)