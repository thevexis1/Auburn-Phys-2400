#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 08:10:34 2025

@author: bradleyandrew
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N = 64           # Number of grid points in each dimension
Lx, Ly = 2*np.pi, 2*np.pi  # Domain size
kappa = 0.1       # Thermal diffusivity
dt = 0.001        # Time step
tmax = 4        # Maximum time
steps = int(tmax / dt)

# Spatial grid
x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
xx, yy = np.meshgrid(x, y)

# Wavenumbers for Fourier differentiation
kx = np.fft.fftfreq(N, d=Lx/(2*np.pi*N))
ky = np.fft.fftfreq(N, d=Ly/(2*np.pi*N))
KX, KY = np.meshgrid(kx, ky)
K_squared = KX**2 + KY**2

# Initial condition: 2D Gaussian
u0 = np.exp(-10 * ((xx - np.pi)**2 + (yy - np.pi)**2))
u_hat = np.fft.fft2(u0)

# Storage for animation
snapshots = [np.real(np.fft.ifft2(u_hat))]

# Time stepping
for step in range(steps):
    # Exact solution in Fourier space
    u_hat = u_hat * np.exp(-kappa * K_squared * dt)
    
    # Save snapshot every 20 steps
    if step % 10 == 0:
        u_real = np.real(np.fft.ifft2(u_hat))
        snapshots.append(u_real)

# ------------------------
# Create animation
# ------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Set up surface plot
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

# -----------------------------
# Parameters
# -----------------------------
n = 20             # Number of interior points per dimension
N = n * n          # Total number of unknowns (flattened 2D -> 1D)
Lx, Ly = 1.0, 1.0  # Physical domain size
kappa = 0.001        # Diffusion coefficient
dt = 0.0005        # Time step
tmax = 4        # Maximum time
steps = int(tmax / dt)

# Grid spacing
hx = Lx / (n + 1)  # distance between grid points (interior only)
hy = Ly / (n + 1)

# Physical grid for plotting
x = np.linspace(hx, Lx-hx, n)  # n points from hx to Lx-hx
y = np.linspace(hy, Ly-hy, n)
xx, yy = np.meshgrid(x, y)

# -----------------------------
# Build the 2D Laplacian matrix
# -----------------------------
# Build 1D second derivative matrix T
I = np.eye(n)
e = np.ones(n)
T = np.diag(-4 * e) + np.diag(e[:-1], k=1) + np.diag(e[:-1], k=-1)

# Kronecker build:
# np.kron(I, T) -> second derivative in x
# np.kron(T, I) -> second derivative in y
L = np.kron(I, T) + np.kron(T, I)

# Scale Laplacian by grid spacing
L = L / (hx ** 2)

# -----------------------------
# Initial condition
# -----------------------------
# Create a 2D Gaussian centered at (0.5, 0.5)
u0 = np.exp(-50 * ((xx - 0.5)**2 + (yy - 0.5)**2))

# Flatten the 2D initial condition into 1D for matrix multiplication
u = u0.flatten()

# -----------------------------
# Storage for animation
# -----------------------------
snapshots = [u.reshape((n, n))]

# -----------------------------
# Time stepping (Forward Euler)
# -----------------------------
for step in range(steps):
    # Forward Euler: u_new = u_old + dt * kappa * Laplacian(u_old)
    u = u + dt * kappa * (L @ u)

    if step % 10 == 0:
        snapshots.append(u.reshape((n, n)))

# -----------------------------
# Create animation
# -----------------------------
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

# -----------------------------------
# Nicer matrix for visualization
# -----------------------------------
L_nicer = L * hx**2  # Rescale so entries are ~-4 and +1

# Take a small top-left subsection
subsection_size = 50  # can adjust
L_sub = L_nicer[:subsection_size, :subsection_size]


# Plot
plt.figure(figsize=(10, 8))
plt.imshow(L_sub, cmap='coolwarm', origin='lower')
plt.colorbar(label='Matrix Value')
plt.title(f'2D Laplacian Matrix (Top {subsection_size}x{subsection_size})')
plt.xlabel('Column')
plt.ylabel('Row')

# Add text numbers inside matrix
for i in range(subsection_size):
    for j in range(subsection_size):
        plt.text(j, i, f"{L_sub[i, j]:.0f}", ha='center', va='center', color='black', fontsize=6)

plt.tight_layout()
plt.show()

