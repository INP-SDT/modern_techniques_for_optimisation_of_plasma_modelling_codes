# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:44:30 2024
@author: WagnerR
Note:
Modified on Tue Nov 05 17:37 2024
by A Jovanovic
"""
from nanomesh import Mesher2D, Image
from skfem import MeshTri
import numpy as np
from PIL import Image as PILI
# import tkinter as tk
# from tkinter import filedialog

# load image
# root = tk.Tk()
# root.withdraw()  # Hide the root window
# filename = filedialog.askopenfilename(title="Select an image")
filename = 'technical_drawing.png'
image = PILI.open(filename)

# Get the size of the image
image_size = image.size
# Extract width and height
image_width = image_size[0]
image_height = image_size[1]

# Convert to a 2D NumPy array
image_array = np.array(image)

# Convert to grayscale if it's a colored image
if image_array.ndim == 3:
    # Average the color channels
    image_array = np.mean(image_array, axis=2)
# Optional: Normalize the array to [0, 1] range
#image_array = image_array / 255.0

image = Image(image_array)
image.show()

# Create mesh based on image
mesher = Mesher2D(image)
# Adjust max_edge_dist
mesher.generate_contour(max_edge_dist=500, precision=1)
# Visualize the contour
mesher.plot_contour()

# Extract contour coordinates
contour = mesher.contour
coordinates = contour.points

# Build a triangular mesh based on the found contours
# qx = quality mesh with angle > x°; ay =  maximum triangle size of y px
mesh = mesher.triangulate(opts='q30a120')

"""
# Define the scaling factor, for bot x- and y-axis
scale_factor = 0.1  # This can be adjusted as needed

# Apply scaling to the mesh vertices (points) while maintaining the aspect ratio
for vertex in mesh.points:
    # Scale x
    vertex[0] *= scale_factor
    # Scale y
    vertex[1] *= scale_factor
"""
"""
# Custom scaling factors for x and y individually
# Scale for x-axis
scale_x = (1/image_width) * 1.0
# Scale for y-axis
scale_y = (1/image_height) * 0.5

# Apply scaling to the mesh vertices
for vertex in mesh.points:
    # Scale x
    vertex[0] *= scale_x
    # Scale y
    vertex[1] *= scale_y
"""

# write mesh into a .msh file
try:
    mesh.write(filename.split('.')[0]+"gmsh22"  + '.msh', file_format='gmsh22', binary=False)
except Exception:
    pass
# try:
#     mesh.write(filename.split('.')[0]+"dolfin-xml"  + '.xml', file_format='dolfin-xml')
# except Exception:
#     pass
# try:
#     mesh.write(filename.split('.')[0]+"xdmf"  + '.xdmf', file_format='xdmf', binary=False)
# except Exception:
#     pass

image.compare_with_mesh(mesh)

triangles = mesh.get('triangle')

#label=1 is for the inner contour mesh
#label=2 is for the outer contour mesh
triangles.remove_cells(label=1, key="physical")

# Plot triangular mesh
triangles.plot()

# Coordinates of the vertices
p = triangles.points.T
# Connectivity of the triangles
t = triangles.cells.T

m = MeshTri(p, t)
print(m)

import meshio

fname = filename.split('.')[0]+"gmsh22"  + '.msh'
mesh = meshio.read(fname)
mesh.write(filename.split('.')[0]+"gmsh22" + '.vtu')
