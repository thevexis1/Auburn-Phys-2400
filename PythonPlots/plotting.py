#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:12:08 2025

@author: bradleyandrew
"""

#%% Line Plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%%
x = np.linspace(-5, 5, 1000)
y = ((np.cos(x))**2)*(np.exp(-x**4))
plt.figure()
plt.plot(x, y, label='Sine Wave')
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid()
plt.show()

#%% Scatter Plot
x = np.random.rand(100)
y = np.random.rand(100)
plt.figure()
plt.scatter(x, y, c='blue', alpha=0.5)
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

#%% Histogram
data = np.random.randn(1000)
plt.figure()
plt.hist(data, bins=30, color='green', alpha=0.7)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

#%% Bar Plot
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 5, 9]
plt.figure()
plt.bar(categories, values, color='orange')
plt.title("Bar Plot")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()

#%% 2D Contour Plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
plt.figure()
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar()
plt.title("2D Contour Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

#%% 3D Surface Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("3D Surface Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()

#%% Vector Field (Quiver Plot)
X, Y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
U = -Y  # x-component of vector
V = X   # y-component of vector
plt.figure()
plt.quiver(X, Y, U, V, color='red')
plt.title("Vector Field (Quiver Plot)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

#%% Pie Chart
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # explode the 2nd slice
plt.figure()
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart")
plt.show()

#%% Box Plot
data = [np.random.rand(50) * i for i in range(1, 6)]
plt.figure()
plt.boxplot(data, vert=True, patch_artist=True, labels=['A', 'B', 'C', 'D', 'E'])
plt.title("Box Plot")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()

#%% Heatmap
data = np.random.rand(10, 10)
plt.figure()
plt.imshow(data, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("Heatmap")
plt.show()
