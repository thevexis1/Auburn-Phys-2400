#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 08:10:34 2025

@author: bradleyandrew
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 64
Lx, Ly = 2*np.pi, 2*np.pi  
kappa = 0.8     
dt = 0.001
tmax = 4        
steps = int(tmax / dt)

x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
xx, yy = np.meshgrid(x, y)

kx = np.fft.fftfreq(N, d=Lx/(2*np.pi*N))
ky = np.fft.fftfreq(N, d=Ly/(2*np.pi*N))
KX, KY = np.meshgrid(kx, ky)
K_squared = KX**2 + KY**2

u0 = np.exp(-10 * ((xx - np.pi)**2 + (yy - np.pi)**2))
u_hat = np.fft.fft2(u0)

snapshots = [np.real(np.fft.ifft2(u_hat))]

for step in range(steps):
    u_hat = u_hat * np.exp(-kappa * K_squared * dt)
    
    if step % 10 == 0:
        u_real = np.real(np.fft.ifft2(u_hat))
        snapshots.append(u_real)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surf = [ax.plot_surface(xx, yy, snapshots[0], cmap='inferno')]

def update(frame):
    ax.clear()
    ax.plot_surface(xx, yy, snapshots[frame], cmap='inferno')
    ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(f"Time step: {frame * 20 * dt:.3f} s")

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=100)

plt.tight_layout()
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

