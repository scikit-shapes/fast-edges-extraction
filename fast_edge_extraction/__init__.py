from ._vtk import edges as edges_vtk  # noqa: F401
from ._torch import edges as edges_torch  # noqa: F401
from ._cython import edges as edges_cython_core
from typing import Tuple
import pyvista
import torch
import numpy as np


def edges_cython(
    points: torch.Tensor, triangles: torch.Tensor
) -> Tuple[np.array, np.array]:
    """Interface to the cython function edges_cython_core"""

    points = points.cpu().numpy()
    # triangles are sorted
    # triangles = triangles.sort(dim=0)[0]
    triangles = triangles.cpu().numpy()

    edges = edges_cython_core(triangles.astype(np.int64))[0]

    return torch.from_numpy(edges).long()


def compute_points_and_triangles(
    mesh: pyvista.PolyData,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the points and triangles of a mesh as torch.Tensor.

    Args:
        mesh (pyvista.PolyData): a mesh

    Returns:
        torch.Tensor: a (n_points, 3) tensor of points
        torch.Tensor: a (3, n_triangles) tensor of triangles
    """

    # remove padding
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    triangles = torch.from_numpy(triangles.copy()).long()

    points = torch.from_numpy(mesh.points).float()

    return points, triangles


def sort_edges(edges: torch.Tensor) -> np.array:
    """Lexicographically sort a (2, n_edges) array of edges to
    allow for comparison with various implementations of edges
    extraction.

    Args:
        edges (torch.Tensor): a (2, n_edges) array of edges

    Returns:
        np.array: the sorted edges
    """
    edges = edges.sort(dim=1)[0].cpu().numpy()
    ordering = np.lexsort((edges[:, 1], edges[:, 0]))
    return edges[ordering]
