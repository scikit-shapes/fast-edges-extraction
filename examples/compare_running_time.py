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
edges_vtk = mesh.extract_all_edges()
end_vtk = time()
time_vtk = end_vtk - start_vtk
print(f"|VTK               | {time_vtk:.3f}" + length_blank(time_vtk) * " " + "|")

# Numba
# ------

start_warmup_numba = time()
edges = fast_edge_extraction.edges_numba(points, triangles)
end_warmup_numba = time()
time_warmup_numba = end_warmup_numba - start_warmup_numba

start_numba = time()
edges_numba = fast_edge_extraction.edges_numba(points, triangles)
end_numba = time()
time_numba = end_numba - start_numba

print(
    f"|VTK+Numba compile | {time_warmup_numba - time_numba:.3f}"
    + length_blank(time_warmup_numba - time_numba) * " "
    + "|"
)


print(f"|VTK+Numba         | {time_numba:.3f}" + length_blank(time_numba) * " " + "|")

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
print(f"Number of edges: {edges_cython.shape[1]}")
print()


print("Checking consistency between implementations...")


def normalize_edges(edges: torch.Tensor) -> np.array:
    """Lexicographically sort a (2, n_edges) array of edges to
    allow for comparison with various implementations of edges
    extraction.

    Args:
        edges (torch.Tensor): a (2, n_edges) array of edges

    Returns:
        np.array: the sorted edges
    """
    edges = edges.sort(dim=0)[0].cpu().numpy()
    ordering = np.lexsort((edges[1], edges[0]))
    return edges[:, ordering]


edges_numba = normalize_edges(edges_numba)
edges_torch = normalize_edges(edges_torch)
edges_cython = normalize_edges(edges_cython)

assert np.allclose(edges_numba, edges_torch)
assert np.allclose(edges_torch, edges_cython)
print("Consistency check passed!")
