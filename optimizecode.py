#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:00:55 2025

@author: bradleyandrew
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count
import psutil


# --------------------------------------------
# Backend System Discussion
# --------------------------------------------

# Spyder Tools:
# - Use `%prun`, `%time`, `%timeit`, `%memit` (from memory_profiler)
# - Memory usage: psutil.virtual_memory().available

print("\nSystem Information:")
print(f"Available CPU cores: {cpu_count()}")
print(f"Total RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

# Notes:
# - Python uses CPython, which executes bytecode line by line in a C-based virtual machine.
# - The naive loop is single-threaded due to the GIL (Global Interpreter Lock).
# - Multiprocessing bypasses the GIL by creating separate processes (not threads).
# - Each worker has its own memory space; hence large arrays may need to be passed efficiently (shared memory or memory mapping).
# - Use NumPy and vectorization for further performance.
# - This gravitational potential example for a large number of particles is memory-bound and CPU-intensive: a perfect fit for parallelization.

# You can visualize CPU usage in Task Manager (Windows) or `htop` (Linux/Mac).
# Use `%mprun -f compute_potential_naive compute_potential_naive(0)` in Spyder for line-by-line memory profiling.
# Use `%prun compute_potential_naive(0)` for function-level timing in IPython.


#%%

from numba import njit
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant
N_PARTICLES = 5000
np.random.seed(42)
positions = np.random.rand(N_PARTICLES, 3) * 1e6
masses = np.random.rand(N_PARTICLES) * 1e3

# NAIVE METHOD
def compute_potential_naive(i):
    phi = 0.0
    for j in range(N_PARTICLES):
        if i != j:
            r = np.linalg.norm(positions[i] - positions[j])
            if r != 0:
                phi -= G * masses[j] / r
    return phi

def run_naive(particles):
    start = time.time()
    result = [compute_potential_naive(i) for i in range(particles)]
    end = time.time()
    print(f"Naive time: {end - start:.2f} s")
    return result, end - start

# MULTIPROCESSING METHOD
def compute_potential_parallel(i):
    phi = 0.0
    for j in range(N_PARTICLES):
        if i != j:
            r = np.linalg.norm(positions[i] - positions[j])
            if r != 0:
                phi -= G * masses[j] / r
    return phi

def run_multiprocessing(particles):
    start = time.time()
    with Pool(cpu_count()) as pool:
        result = pool.map(compute_potential_parallel, range(particles))
    end = time.time()
    print(f"Multiprocessing time: {end - start:.2f} s")
    return result, end - start

# NUMBA METHOD
@njit
def compute_potential_numba(i, positions, masses):
    phi = 0.0
    for j in range(len(masses)):
        if i != j:
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            if r != 0:
                phi -= G * masses[j] / r
    return phi

def run_numba(particles):
    start = time.time()
    result = [compute_potential_numba(i, positions, masses) for i in range(particles)]
    end = time.time()
    print(f"Numba JIT time: {end - start:.2f} s")
    return result, end - start

# KD-TREE METHOD
def run_kdtree(particles, k=50):
    start = time.time()
    tree = cKDTree(positions)
    neighbors = tree.query(positions[:particles], k=k+1)[1]
    potentials = np.zeros(particles)

    for i in range(particles):
        phi = 0.0
        for j in neighbors[i][1:]:
            r = np.linalg.norm(positions[i] - positions[j])
            if r != 0:
                phi -= G * masses[j] / r
        potentials[i] = phi
    end = time.time()
    print(f"KD-Tree time (k={k}): {end - start:.2f} s")
    return potentials, end - start

#%% MAIN EXECUTION AND PLOTTING
if __name__ == "__main__":
    particles = 800

    naive, t_naive = run_naive(particles)
    parallel, t_parallel = run_multiprocessing(particles)
    numba, t_numba = run_numba(particles)
    kdtree, t_kdtree = run_kdtree(particles, k=50)

    diff_np = np.abs(np.array(naive) - np.array(parallel)).max()
    diff_nn = np.abs(np.array(naive) - np.array(numba)).max()
    diff_kd = np.abs(np.array(naive) - np.array(kdtree)).max()

    print(f"\nMax difference (Naive vs Parallel): {diff_np:.6e}")
    print(f"Max difference (Naive vs Numba): {diff_nn:.6e}")
    print(f"Max difference (Naive vs KDTree): {diff_kd:.6e}")
    print(f"\nSpeedup Parallel: {t_naive / t_parallel:.2f}x")
    print(f"Speedup Numba: {t_naive / t_numba:.2f}x")
    print(f"Speedup KDTree: {t_naive / t_kdtree:.2f}x")

    # PLOT
    methods = ['Naive', 'Multiprocessing', 'Numba', 'KDTree']
    times = [t_naive, t_parallel, t_numba, t_kdtree]

    plt.figure(figsize=(10, 6))
    plt.bar(methods, times, color='skyblue', edgecolor='black')
    plt.ylabel("Execution Time (s)")
    plt.title(f"Performance Comparison for {particles} Particles")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


#%%

import io
import pstats
from IPython.core.magics.execution import _format_time
from memory_profiler import memory_usage
import sys
from contextlib import redirect_stdout

output_file = "optstats"

s = io.StringIO()
%prun -q -s cumulative -T optstats run_naive(100)
print(s.getvalue())


# Redirect stdout to file
with open(output_file, "a") as f:  # Use "a" to append if prun output is already there
    with redirect_stdout(f):
        # Print a header so it's readable
        print("\n\n===== Memory Profile: compute_potential_naive =====\n")
        %load_ext memory_profiler
        %mprun -f compute_potential_naive compute_potential_naive(0)
        
# Manually run mprun and copy/paste output to the file
# (IPython does not support redirecting mprun to file programmatically)
        
