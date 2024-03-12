from math import log10
from time import time

import numpy as np

import fast_edges_extraction

try:
    import pyvista as pv
    PV_AVAILABLE = True

    def edges_vtk(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
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
        # Create a pyvista mesh
        mesh = pv.PolyData.from_regular_faces(points, triangles)
        # Use VTK method to extract the edges
        wireframe = mesh.extract_all_edges(use_all_points=True)
        # Extract the edges
        return wireframe.lines.reshape(-1, 3)[:, 1:]

except ImportError:
    PV_AVAILABLE = False

def sort_edges(edges: np.ndarray) -> np.ndarray:
    """Lexicographically sort a (n_edges, 2) array of edges to
    allow for comparison with various implementations of edges
    extraction.

    Args:
        edges (torch.Tensor): a (n_edges, 2) array of edges

    Returns:
        np.array: the sorted edges
    """
    assert edges.shape[1] == 2
    edges.sort(axis=1)
    ordering = np.lexsort((edges[:, 1], edges[:, 0]))
    return edges[ordering]

def edges_numpy(triangles: np.ndarray) -> np.ndarray:
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


if PV_AVAILABLE:
    from pyvista import examples
    mesh = examples.download_bunny()
    points = mesh.points
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]

else:

    generator = np.random.default_rng(0)
    points = generator.random((1000, 3))
    triangles = generator.integers(0, 1000, (100000, 3))

print()
print(f"{len(points)} points, {len(triangles)} triangles")
print("-----------------------------------")
print("|Implementation    | Running time |")
print("|------------------+---------------")


def length_blank(t: float) -> int:
    """compute the length of the blank space to display the table"""
    return 8 - int(log10(max([t, 1])))


if PV_AVAILABLE:
    # VTK
    # ----

    start_vtk = time()
    edges_vtk = edges_vtk(points, triangles)
    end_vtk = time()
    time_vtk = end_vtk - start_vtk
    print(f"|VTK               | {time_vtk:.3f}" + length_blank(time_vtk) * " " + "|")

# Torch
# ------

start_np = time()
edges_np = edges_numpy(triangles)
end_np = time()
time_np = end_np - start_np
print(f"|Numpy             | {time_np:.3f}" + length_blank(time_np) * " " + "|")

# Cython
# -------

start_cython = time()
edges_cython = fast_edges_extraction.extract_edges(triangles)
end_cython = time()
time_cython = end_cython - start_cython
print(f"|Cython            | {time_cython:.3f}" + length_blank(time_cython) * " " + "|")

print("----------------------------------")
print(f"Number of edges: {edges_cython.shape[0]}")
print()


print("Checking consistency between implementations...")


edges_torch = sort_edges(edges_np)
edges_cython = sort_edges(edges_cython)
assert np.allclose(edges_torch, edges_cython)

if PV_AVAILABLE:
    edges_vtk = sort_edges(edges_vtk)
    assert np.allclose(edges_torch, edges_vtk)

print("Consistency check passed!")
