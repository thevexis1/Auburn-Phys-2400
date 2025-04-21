#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 08:10:34 2025

@author: bradleyandrew
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 64 #changes the amplitude 
Lx, Ly = 2*np.pi, 2*np.pi  #where it lays within the 2-D coordinate system
kappa = 0.1 #changes the rate of dissipation    
dt = 0.001 #changes the rate of diispation with respect to time
tmax = 0.1 #sets the max temperature range of change       
steps = int(tmax / dt) #integrates the temperature with respect to time 

#sets the boundaries of the coordinate plane
x = np.linspace(0, Lx, N, endpoint=False) #defines the x coordinate axis
y = np.linspace(0, Ly, N, endpoint=False) #defines the y coordinate axis
xx, yy = np.meshgrid(x, y) #creates a grid to represent the diffusion 

kx = np.fft.fftfreq(N, d=Lx/(2*np.pi*N)) #defines the changes of the frequency of the fourier transform in the x direction 
ky = np.fft.fftfreq(N, d=Ly/(2*np.pi*N)) #defines the changes of frequency of the fourier transform in the y direction 
KX, KY = np.meshgrid(kx*8, ky) #creates a grid from the above values as increments of the fourier transformations
K_squared = KX**2 + KY**2

u0 = np.exp(-10 * ((xx - np.pi)**2 + (yy - np.pi)**2)) #diffusion function/equation
u_hat = np.fft.fft2(u0) #creates a fourier transform of the diffusuon equation 

snapshots = [np.real(np.fft.ifft2(u_hat))] #shows a snapshot of the diffusion at it max before dissipation

for step in range(steps): #this for loop integrates all of the steps together
    u_hat = u_hat * np.exp(-kappa * K_squared * dt)
    
    if step % 10 == 0: #if the step is divided by 10 and has a remainder it sets it equal to 1
        u_real = np.real(np.fft.ifft2(u_hat))
        snapshots.append(u_real) #this simplifies the dissipation to create a better snapshot

fig = plt.figure(figsize=(8, 6)) #plots the figure
ax = fig.add_subplot(111, projection='3d') #creates a 3d figure 

surf = [ax.plot_surface(xx, yy, snapshots[0], cmap='inferno')] #adds plot for the grids

def update(frame): #Sets labels for plots and updates each frame 
    ax.clear()
    ax.plot_surface(xx, yy, snapshots[frame], cmap='inferno')
    ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(f"Time step: {frame * 20 * dt:.3f} s")

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=100) #animates the 3d figure 

plt.tight_layout() #prints plots with a neat layout
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 20           
N = n * n         
Lx, Ly = 1.0, 1.0  
kappa = 0.001       
dt = 0.0005       
tmax = 4    
steps = int(tmax / dt)

hx = Lx / (n + 1)  
hy = Ly / (n + 1)

x = np.linspace(hx, Lx-hx, n)
y = np.linspace(hy, Ly-hy, n)
xx, yy = np.meshgrid(x, y)

I = np.eye(n)
e = np.ones(n)
T = np.diag(-4 * e) + np.diag(e[:-1], k=1) + np.diag(e[:-1], k=-1)

L = np.kron(I, T) + np.kron(T, I)

L = L / (hx ** 2)

u0 = np.exp(-50 * ((xx - 0.5)**2 + (yy - 0.5)**2))

u = u0.flatten()

snapshots = [u.reshape((n, n))]

for step in range(steps):
    u = u + dt * kappa * (L @ u)

    if step % 10 == 0:
        snapshots.append(u.reshape((n, n)))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    ax.plot_surface(xx, yy, snapshots[frame], cmap='inferno')
    ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(f"Time: {frame * 20 * dt:.4f} s")

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=100)

plt.tight_layout()
plt.show()


L_nicer = L * hx**2  

subsection_size = 50  
L_sub = L_nicer[:subsection_size, :subsection_size]

plt.figure(figsize=(10, 8))
plt.imshow(L_sub, cmap='coolwarm', origin='lower')
plt.colorbar(label='Matrix Value')
plt.title(f'2D Laplacian Matrix (Top {subsection_size}x{subsection_size})')
plt.xlabel('Column')
plt.ylabel('Row')

for i in range(subsection_size):
    for j in range(subsection_size):
        plt.text(j, i, f"{L_sub[i, j]:.0f}", ha='center', va='center', color='black', fontsize=6)

plt.tight_layout()
plt.show()

