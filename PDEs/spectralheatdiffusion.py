#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 08:10:34 2025

@author: bradleyandrew
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
"""This section below defines the values of the variables 
used in the equation and defines the intervals at which x and y will decrease.
The N is the grid size and the Lx and Ly is the domain size."""
N = 64
Lx, Ly = 2*np.pi, 2*np.pi  
kappa = 0.1     
dt = 0.001
tmax = 4        
steps = int(tmax / dt)
"""The linspace below is used for the numerical computation and 
the meshgrid is used to make a grid from the computed x and y variables."""
x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
xx, yy = np.meshgrid(x, y)
"""The fft is being used to compute the fast Fourier Transform of kappa x and y, 
the meshgrid is used to make a grid for the computed kappa x and y, 
and the K_squared is used to help compute the Laplacian in the Fast Fourier Transform.""" 
kx = np.fft.fftfreq(N, d=Lx/(2*np.pi*N))
ky = np.fft.fftfreq(N, d=Ly/(2*np.pi*N))
KX, KY = np.meshgrid(kx, ky)
K_squared = KX**2 + KY**2
"""The u0 takes the x and y computed above and puts it into an exponent and the u_hat takes the fast Fourier
Transform of the computed u0."""
u0 = np.exp(-10 * ((xx - np.pi)**2 + (yy - np.pi)**2))
u_hat = np.fft.fft2(u0)
"""The snapshot takes a picture of the object in its current state and compares it to the previous snapshot."""
snapshots = [np.real(np.fft.ifft2(u_hat))]
"""This is a loop that takes every 10th snapshot and plots it using everything else below and taking the 
integration and then comes back to this and takes 
the snapshot of the 10th interval again and runs it through everything below."""
for step in range(steps):
    u_hat = u_hat * np.exp(-kappa * K_squared * dt)
    
    if step % 10 == 0:
        u_real = np.real(np.fft.ifft2(u_hat))
        snapshots.append(u_real)
"""This plots the figure and the size of the figure and the subplot is redundant in this."""
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
"""This is used to make the 3D figure shown on the graph"""
surf = [ax.plot_surface(xx, yy, snapshots[0], cmap='inferno')]
"""This is used for the actual animation of the graph by taking the snapshots and putting them in sequential
order so that the graph looks animated. It also labels the axis and creates the timer displayed at the 
top of the graph."""
def update(frame):
    ax.clear()
    ax.plot_surface(xx, yy, snapshots[frame], cmap='inferno')
    ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(f"Time step: {frame * 20 * dt:.3f} s")
"""This is used to plot the above and how fast the animation occurs."""
ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=100)
"""This is just to actually plot the graph."""
plt.tight_layout()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
"""This is the same thing as the one in part 1 except with a smaller grid size."""
n = 20           
N = n * n         
Lx, Ly = 1.0, 1.0  
kappa = 0.001       
dt = 0.0005       
tmax = 4    
steps = int(tmax / dt)
"""These are used to calculate the grid space."""
hx = Lx / (n + 1)  
hy = Ly / (n + 1)
"""These are the boundary conditions for the object/grid."""
x = np.linspace(hx, Lx-hx, n)
y = np.linspace(hy, Ly-hy, n)
xx, yy = np.meshgrid(x, y)
"""The I creates an identity matrix of size nxn. The e creates a vector array filled with 1s for whatever value 
n is. The T is used to create the matrix from the values of n given in I and e."""
I = np.eye(n)
e = np.ones(n)
T = np.diag(-4 * e) + np.diag(e[:-1], k=1) + np.diag(e[:-1], k=-1)
"""This is used to create a Laplacian matrix. The (I,T) is the matrix in the x direction and the (T,I) 
is the matrix in the y direction. The division using h^2 is used to correctly scale the Laplacian."""
L = np.kron(I, T) + np.kron(T, I)

L = L / (hx ** 2)
"""This is a Gausian centered in the middle of the domain. The flatten turns u0 into a vector to match the 
matrix above."""
u0 = np.exp(-50 * ((xx - 0.5)**2 + (yy - 0.5)**2))

u = u0.flatten()
"""This is used like in part 1 and just takes pictures of the object so it can then compare it with previous 
snapshots to help make the animation."""
snapshots = [u.reshape((n, n))]
"""This is almost exactly the same thing as in part 1 but it uses Euler's formula instead of integration. """
for step in range(steps):
    u = u + dt * kappa * (L @ u)

    if step % 10 == 0:
        snapshots.append(u.reshape((n, n)))
"""This is used to plot the figure on the grid."""
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
"""This is the same as in part 1 and is used for the animation of the figure in the grid."""
def update(frame):
    ax.clear()
    ax.plot_surface(xx, yy, snapshots[frame], cmap='inferno')
    ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(f"Time: {frame * 20 * dt:.4f} s")
"""This is the line of code that actually animates the figure and the interval sets the delay between the snapshots
and the delay is 100ms."""
ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=100)
"""This just plots everything above."""
plt.tight_layout()
plt.show()

"""This rescales the matrix so it's easier to interpret and makes it look better."""
L_nicer = L * hx**2  
"""This grabs a 50x50 section of the matrix and zooms in on it as the original matrix is very big,
so this is used to get a better view of the actual matrix and it's not filled with clutter."""
subsection_size = 50  
L_sub = L_nicer[:subsection_size, :subsection_size]
"""This displays the 2D Laplacian figure. It defines the figure size as well as its color. It also displays the 
labels on the graph. """
plt.figure(figsize=(10, 8))
plt.imshow(L_sub, cmap='coolwarm', origin='lower')
plt.colorbar(label='Matrix Value')
plt.title(f'2D Laplacian Matrix (Top {subsection_size}x{subsection_size})')
plt.xlabel('Column')
plt.ylabel('Row')
"""This is a loop that adds a number on each individual square of the 2D Laplacian figure. The i is the loop for 
rows and the j is the loop for columns. Each pair of (i,j) corresponds to the index in the matrix. The plt.text
is used to place the number in the center of the squares/tiles."""
for i in range(subsection_size):
    for j in range(subsection_size):
        plt.text(j, i, f"{L_sub[i, j]:.0f}", ha='center', va='center', color='black', fontsize=6)
"""This is used to display the 2D Laplacian Figure."""
plt.tight_layout()
plt.show()

