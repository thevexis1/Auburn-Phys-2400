#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:09:51 2025

@author: lochstu
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def mandelbrot(c, max_iter):
    """
    Determine if a point c in the complex plane is in the Mandelbrot set.

    Parameters:
        c (complex): A complex number representing a point in the complex plane.
        max_iter (int): The maximum number of iterations to determine divergence.

    Returns:
        int: The number of iterations before divergence, or max_iter if the point does not diverge.
    """
    z = 0
    for i in range(max_iter):
        if abs(z) > 2:
            return i
        z = z * z + c
    return max_iter

def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generate the Mandelbrot set.

    Parameters:
        xmin, xmax, ymin, ymax (float): Bounds of the complex plane.
        width, height (int): Dimensions of the output image.
        max_iter (int): The maximum number of iterations.

    Returns:
        np.ndarray: A 2D array of iteration counts.
    """
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    mandelbrot_set = np.empty((height, width))

    for i, re in enumerate(real):
        # print('i=',i)
        for j, im in enumerate(imag):
            c = complex(re, im)
            mandelbrot_set[j, i] = mandelbrot(c, max_iter)

    return mandelbrot_set

def plot_mandelbrot(mandelbrot_set, xmin, xmax, ymin, ymax):
    """
    Plot the Mandelbrot set.

    Parameters:
        mandelbrot_set (np.ndarray): A 2D array of iteration counts.
        xmin, xmax, ymin, ymax (float): Bounds of the complex plane.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(mandelbrot_set, extent=[xmin, xmax, ymin, ymax], cmap="hot", origin="lower")
    plt.colorbar(label="Iteration count")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.title("Mandelbrot Set")
    plt.show()

# Parameters
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
width, height = 1600, 1600
max_iter = 50

start_time = time.time()

# Generate and plot the Mandelbrot set
mandelbrot_set = generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
plot_mandelbrot(mandelbrot_set, xmin, xmax, ymin, ymax)


