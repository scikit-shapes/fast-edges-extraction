import fast_edge_extraction
from pyvista import examples
import numpy as np
from time import time

mesh = examples.download_louis_louvre().clean()
points, triangles = fast_edge_extraction.compute_points_and_triangles(mesh)

# Numba
#------

start_warmup_numba = time()
edges = fast_edge_extraction.edges_numba(points, triangles)
end_warmup_numba = time()
print("Numba warmup time: ", end_warmup_numba - start_warmup_numba)

start_numba = time()
edges_numba = fast_edge_extraction.edges_numba(points, triangles)
end_numba = time()
print("Numba time: ", end_numba - start_numba)

# Torch
#------

start_torch = time()
edges_torch = fast_edge_extraction.edges_torch(points, triangles)
end_torch = time()
print("Torch time: ", end_torch - start_torch)


