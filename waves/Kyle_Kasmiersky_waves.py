# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:16:37 2025

@author: kylek
"""

#%% import dependancies
import numpy as np
import matplotlib.pyplot as plt

#%%

# initialize variables

A = np.array([1, 5, -3, 2.5])
omega = np.array([2*np.pi,  4*np.pi, np.pi, np.pi/2])
phi = np.array([-np.pi, np.pi, 0, np.pi*3/4])
Y = [];

# calculate functions | A*e^(-i(wt+phi))

t = np.linspace(0, 2, 1000)

Y0 = A[0]*np.exp(complex(0,-1)*(omega[0]*t + phi[0]))
Y1 = A[1]*np.exp(complex(0,-1)*(omega[1]*t + phi[1]))
Y2 = A[2]*np.exp(complex(0,-1)*(omega[2]*t + phi[2]))
Y3 = A[3]*np.exp(complex(0,-1)*(omega[3]*t + phi[3]))

# create sum of 4 waves

Ysum = Y0 + Y1 + Y2 + Y3



#%%plotting

#Plot first graph, seperate real and imaginary wave

plt.subplot(1,2,1)
plt.plot(t,np.real(Y0))

plt.xlabel("Time")
plt.ylabel("Real")

plt.subplot(1,2,2)
plt.plot(t,np.imag(Y0))

plt.xlabel("Time")
plt.ylabel("Imaginary")

plt.suptitle("Plane wave 1")

plt.show()

#Plot second graph, seperate real and imaginary wave

plt.subplot(1,2,1)
plt.plot(t,np.real(Y1))

plt.xlabel("Time")
plt.ylabel("Real")

plt.subplot(1,2,2)
plt.plot(t,np.imag(Y1))

plt.xlabel("Time")
plt.ylabel("Imaginary")

plt.suptitle("Plane wave 2")

plt.show()

#Plot third graph, seperate real and imaginary wave

plt.subplot(1,2,1)
plt.plot(t,np.real(Y2))

plt.xlabel("Time")
plt.ylabel("Real")

plt.subplot(1,2,2)
plt.plot(t,np.imag(Y2))

plt.xlabel("Time")
plt.ylabel("Imaginary")

plt.suptitle("Plane wave 3")

plt.show()

#Plot fourth graph, seperate real and imaginary wave

plt.subplot(1,2,1)
plt.plot(t,np.real(Y3))

plt.xlabel("Time")
plt.ylabel("Real")

plt.subplot(1,2,2)
plt.plot(t,np.imag(Y3))

plt.xlabel("Time")
plt.ylabel("Imaginary")

plt.suptitle("Plane wave 4")

plt.show()

#Plot fifth graph, a sum of the previous 4
# seperate real and imaginary wave

plt.subplot(1,2,1)
plt.plot(t,np.real(Ysum))

plt.xlabel("Time")
plt.ylabel("Real")

plt.subplot(1,2,2)
plt.plot(t,np.imag(Ysum))

plt.xlabel("Time")
plt.ylabel("Imaginary")

plt.suptitle("Sum of Plane waves")

plt.show()