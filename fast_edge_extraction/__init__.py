from typing import Tuple

import numpy as np
import pyvista

from ._cython import edges as edges_cython_core


def extract_edges(
    points: np.ndarray, triangles: np.ndarray # noqa: ARG001
) -> Tuple[np.ndarray, np.ndarray]:
    """Interface to the cython function edges_cython_core"""
    if triangles.shape[1] != 3:
        msg = "Triangles should have shape (n_triangles, 3)"
        raise ValueError(msg)
    return edges_cython_core(triangles.astype(np.int64))[0]


def compute_points_and_triangles(
    mesh: pyvista.PolyData,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the points and triangles of a mesh as torch.Tensor.

    Args:
        mesh (pyvista.PolyData): a mesh

    Returns:
        torch.Tensor: a (n_points, 3) tensor of points
        torch.Tensor: a (n_triangles, 3) tensor of triangles
    """
    # remove padding
    if not mesh.is_all_triangles:
        msg = "Mesh is not all triangles"
        raise ValueError(msg)

    triangles = np.asarray(mesh.faces.reshape(-1, 4)[:, 1:])
    points = np.asarray(mesh.points)
    return points, triangles


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
