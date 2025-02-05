#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:37:57 2025

@author: bradleyandrew
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import jax
import jax.numpy as jnp

# Cartesian to Cylindrical
def cartesian_to_cylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta, z

def cylindrical_to_cartesian(r, theta, z):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z

# Cartesian to Spherical
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Dipole Coordinates
def cartesian_to_dipole(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta_d = np.arccos(z / r)
    phi = np.arctan2(y, x)
    L = r / np.sin(theta_d)**2
    return L, theta_d, phi

# Helical Coordinates
def cartesian_to_helical(x, y, z, pitch=1):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    h = z - pitch * theta
    return r, theta, h

# Jacobian Calculation
def jacobian_cartesian_to_spherical(x, y, z):
    r, theta, phi = cartesian_to_spherical(x, y, z)
    J = np.array([
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        [-np.sin(phi), np.cos(phi), 0]
    ])
    return J

# Example Demonstrations
if __name__ == "__main__":
    x, y, z = 1, 1, 1

    # Cartesian to Cylindrical
    cylindrical = cartesian_to_cylindrical(x, y, z)
    print(f"Cartesian (1,1,1) → Cylindrical: {cylindrical}")

    # Cartesian to Spherical
    spherical = cartesian_to_spherical(x, y, z)
    print(f"Cartesian (1,1,1) → Spherical: {spherical}")

    # Dipole Conversion
    dipole = cartesian_to_dipole(x, y, z)
    print(f"Cartesian (1,1,1) → Dipole: {dipole}")

    # Helical Conversion
    helical = cartesian_to_helical(x, y, z)
    print(f"Cartesian (1,1,1) → Helical: {helical}")

    # Jacobian Calculation
    J = jacobian_cartesian_to_spherical(x, y, z)
    print(f"Jacobian Matrix for Cartesian → Spherical at (1,1,1):\n{J}")

#%%

# Function to plot Cartesian coordinate system
def plot_cartesian(ax):
    """
    Plots the Cartesian coordinate system with a volume element.
    """
    ax.set_title("Cartesian Volume Element")

    # Cube coordinates
    X = [0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5]
    Y = [0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0]
    Z = [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]

    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]

    for edge in edges:
        ax.plot([X[edge[0]], X[edge[1]]], 
                [Y[edge[0]], Y[edge[1]]], 
                [Z[edge[0]], Z[edge[1]]], 'gray')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Function to plot Spherical coordinate system
def plot_spherical(ax, r=0.5, dtheta=np.pi/4, dphi=np.pi/4):
    """
    Plots the Spherical coordinate system with an appropriately scaled volume element.
    """
    ax.set_title("Spherical Volume Element")

    phi = np.linspace(0, dphi, 30)
    theta = np.linspace(0, dtheta, 30)
    Phi, Theta = np.meshgrid(phi, theta)
    
    X = r * np.sin(Theta) * np.cos(Phi)
    Y = r * np.sin(Theta) * np.sin(Phi)
    Z = r * np.cos(Theta)

    ax.plot_surface(X, Y, Z, color='c', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Function to plot the Jacobian matrix
def plot_jacobian(ax, J):
    """
    Displays the Jacobian matrix as a heatmap.
    """
    ax.set_title("Jacobian Matrix")
    ax.imshow(J, cmap='coolwarm', interpolation='nearest')

    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            ax.text(j, i, f"{J[i, j]:.2f}", ha='center', va='center', color='black')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["dx", "dy", "dz"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["dr", "dθ", "dφ"])

# Compute Jacobian for Cartesian → Spherical
def jacobian_cartesian_to_spherical(x, y, z):
    """
    Computes the Jacobian matrix for Cartesian to Spherical transformation.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    J = np.array([
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        [-np.sin(phi), np.cos(phi), 0]
    ])

    return J

# Generate combined figure for transformation
def visualize_transformation():
    """
    Plots Cartesian and Spherical volume elements with Jacobian in between.
    """
    fig = plt.figure(figsize=(12, 4))

    # Cartesian Volume Element
    ax1 = fig.add_subplot(131, projection='3d')
    plot_cartesian(ax1)

    # Jacobian Matrix
    ax2 = fig.add_subplot(132)
    J = jacobian_cartesian_to_spherical(1, 1, 1)
    plot_jacobian(ax2, J)

    # Spherical Volume Element
    ax3 = fig.add_subplot(133, projection='3d')
    plot_spherical(ax3)

    plt.tight_layout()
    plt.show()

# Run the visualization
visualize_transformation()


#%%

def plot_cylindrical(ax):
    ax.set_title("Cylindrical Coordinates")
    theta = np.linspace(0, 2 * np.pi, 30)
    z = np.linspace(0, 1, 10)
    Theta, Z = np.meshgrid(theta, z)
    R = 0.5
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    ax.plot_surface(X, Y, Z, color='orange', alpha=0.3)
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_toroidal(ax):
    ax.set_title("Toroidal Coordinates")
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    R = 1
    r = 0.5
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, color='purple', alpha=0.3)
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def compute_jacobian_sympy(r_val=0.5, theta_val=sp.pi/4, z_val=0.5):
    r, theta, z = sp.symbols('r theta z')
    R0 = 1
    R = R0 + r * sp.cos(theta)
    Theta = theta
    Zeta = z
    toroidal_coords = sp.Matrix([R, Theta, Zeta])
    cylindrical_coords = sp.Matrix([r, theta, z])
    J = toroidal_coords.jacobian(cylindrical_coords)
    substitutions = {r: r_val, theta: theta_val, z: z_val}
    return J.subs(substitutions)

def cylindrical_to_toroidal(cyl_coords):
    r, theta, z = cyl_coords
    R0 = 1
    R = R0 + r * jnp.cos(theta)
    Theta = theta
    Zeta = z
    return jnp.array([R, Theta, Zeta])

def compute_jacobian_jax(coords):
    return jax.jacfwd(cylindrical_to_toroidal)(coords)

def plot_jacobian_heatmap(ax, J, title="Jacobian (JAX)"):
    ax.set_title(title)
    heatmap = ax.imshow(J, cmap='coolwarm', interpolation='nearest')
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            ax.text(j, i, f"{J[i, j]:.2f}", ha='center', va='center', color='black')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["dr", "dθ", "dz"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["dR", "dΘ", "dζ"])
    plt.colorbar(heatmap, ax=ax)

def plot_jacobian_matrix(ax, J, title="Jacobian (SymPy)"):
    ax.set_title(title)
    ax.axis('off')
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            ax.text(j, i, str(J[i, j]), ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
    ax.set_xticks(np.arange(-0.5, J.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, J.shape[0], 1), minor=True)
    ax.grid(which="minor", color='black', linestyle='-', linewidth=1)

def visualize_coordinate_systems():
    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(141, projection='3d')
    plot_cylindrical(ax1)
    ax2 = fig.add_subplot(142)
    J_sympy = compute_jacobian_sympy()
    plot_jacobian_matrix(ax2, J_sympy, title="Jacobian (SymPy)")
    ax3 = fig.add_subplot(143)
    coords = jnp.array([0.5, jnp.pi / 4, 0.5])
    J_jax = compute_jacobian_jax(coords)
    plot_jacobian_heatmap(ax3, J_jax, title="Jacobian (JAX)")
    ax4 = fig.add_subplot(144, projection='3d')
    plot_toroidal(ax4)
    plt.tight_layout()
    plt.show()

visualize_coordinate_systems()
