from math import log10
from time import time

import numpy as np

import fast_edge_extraction

# VTK and torch implementations

def edges_numpy(
        points: np.ndarray, triangles: np.ndarray # noqa: ARG001
    ) -> np.ndarray:
    """Return the edges of the mesh

    Parameters
    ----------
    points
        the points of the mesh
    triangle
        the triangles of the mesh

    Returns
    -------
        edges (|E|x2 torch.Tensor): the edges of the mesh
    """
    assert triangles.shape[1] == 3
    # Compute the edges of the triangles and sort them
    repeated_edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [0, 2]],
        ],
        axis=0,
    )
    repeated_edges.sort(axis=1)
    return np.unique(repeated_edges, axis=0)



# mesh = examples.download_louis_louvre().clean()
# points, triangles = fast_edge_extraction.compute_points_and_triangles(mesh)

generator = np.random.default_rng(0)
points = generator.random((1000, 3))
triangles = generator.integers(0, 50000, (50000, 3))

print()
print(f"{len(points)} points, {len(triangles)} triangles")
print("-----------------------------------")
print("|Implementation    | Running time |")
print("|------------------+---------------")


def length_blank(t: float) -> int:
    """compute the length of the blank space to display fancy table"""
    return 8 - int(log10(max([t, 1])))


# # VTK
# # ----

# start_vtk = time()
# edges_vtk = edges_vtk(points, triangles)
# end_vtk = time()
# time_vtk = end_vtk - start_vtk
# print(f"|VTK               | {time_vtk:.3f}" + length_blank(time_vtk) * " " + "|")

# Torch
# ------

start_np = time()
edges_np = edges_numpy(points, triangles)
end_np = time()
time_np = end_np - start_np
print(f"|Numpy             | {time_np:.3f}" + length_blank(time_np) * " " + "|")

# Cython
# -------

start_cython = time()
edges_cython = fast_edge_extraction.extract_edges(points, triangles)
end_cython = time()
time_cython = end_cython - start_cython
print(f"|Cython            | {time_cython:.3f}" + length_blank(time_cython) * " " + "|")

print("----------------------------------")
print(f"Number of edges: {edges_cython.shape[0]}")
print()


print("Checking consistency between implementations...")


edges_torch = fast_edge_extraction.sort_edges(edges_np)
edges_cython = fast_edge_extraction.sort_edges(edges_cython)
# edges_vtk = fast_edge_extraction.sort_edges(edges_vtk)

assert np.allclose(edges_torch, edges_cython)
# assert np.allclose(edges_torch, edges_vtk)
print("Consistency check passed!")
