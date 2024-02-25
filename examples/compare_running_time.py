import fast_edge_extraction
from pyvista import examples
import pyvista as pv
import numpy as np
import torch
from time import time
from math import log10

# VTK and torch implementations

def edges_torch(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
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
    repeated_edges = torch.concat(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [0, 2]],
        ],
        dim=0,
    ).sort(dim=1)[0]
    edges = torch.unique(repeated_edges, dim=0)
    return edges


def edges_vtk(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Return the edges of the mesh

    Parameters
    ----------
    points
        the points of the mesh
    triangles
        the triangles of the mesh

    Returns
    -------
        edges (|E|x2 torch.Tensor): the edges of the mesh
    """

    assert triangles.shape[1] == 3

    faces = triangles.clone().cpu().numpy()
    points = points.cpu().numpy()
    mesh = pv.PolyData.from_regular_faces(points, faces)

    wireframe = mesh.extract_all_edges(use_all_points=True)
    edges = wireframe.lines.reshape(-1, 3)[:, 1:]
    return torch.from_numpy(edges)


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
edges_vtk = edges_vtk(points, triangles)
end_vtk = time()
time_vtk = end_vtk - start_vtk
print(f"|VTK               | {time_vtk:.3f}" + length_blank(time_vtk) * " " + "|")

# Torch
# ------

start_torch = time()
edges_torch = edges_torch(points, triangles)
end_torch = time()
time_torch = end_torch - start_torch
print(f"|Torch             | {time_torch:.3f}" + length_blank(time_torch) * " " + "|")

# Cython
# -------

start_cython = time()
edges_cython = edges_cython = fast_edge_extraction.extract_edges(points, triangles)
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
