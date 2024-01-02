import fast_edge_extraction
from pyvista import examples
import pyvista
import numpy as np
import torch
from time import time
from math import log10


mesh = examples.download_louis_louvre().clean()
points, triangles = fast_edge_extraction.compute_points_and_triangles(mesh)

print()
print(f"{mesh.n_points} points, {mesh.n_cells} triangles")
print("-----------------------------------")
print("|Implementation    | Running time |")
print("|------------------+---------------")


def length_blank(t: float) -> int:
    """compute the length of the blank space to display fancy table"""
    return 8 - int(log10(max([t, 1])))


# VTK
# ----

start_vtk = time()
edges_vtk = fast_edge_extraction.edges_vtk(points, triangles)
end_vtk = time()
time_vtk = end_vtk - start_vtk
print(f"|VTK               | {time_vtk:.3f}" + length_blank(time_vtk) * " " + "|")

# Torch
# ------

start_torch = time()
edges_torch = fast_edge_extraction.edges_torch(points, triangles)
end_torch = time()
time_torch = end_torch - start_torch
print(f"|Torch             | {time_torch:.3f}" + length_blank(time_torch) * " " + "|")

# Cython
# -------

start_cython = time()
edges_cython = edges_cython = fast_edge_extraction.edges_cython(points, triangles)
end_cython = time()
time_cython = end_cython - start_cython
print(f"|Cython            | {time_cython:.3f}" + length_blank(time_cython) * " " + "|")

print("----------------------------------")
print(f"Number of edges: {edges_cython.shape[0]}")
print()


print("Checking consistency between implementations...")


edges_torch = fast_edge_extraction.sort_edges(edges_torch)
edges_cython = fast_edge_extraction.sort_edges(edges_cython)
edges_vtk = fast_edge_extraction.sort_edges(edges_vtk)

assert np.allclose(edges_torch, edges_cython)
assert np.allclose(edges_torch, edges_vtk)
print("Consistency check passed!")
