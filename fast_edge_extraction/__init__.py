from ._numba import edges as edges_numba
from ._torch import edges as edges_torch
from ._cython import edges as edges_cython_core
from typing import Tuple
import pyvista
import torch
from typing import Tuple
import numpy as np


def compute_edges(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Compute the edges of a triangle mesh.

    Args:
        points (torch.Tensor): (n_points, 3) float32 tensor of points
        triangles (torch.Tensor): (3, n_triangles) int64 tensor of triangles

    Returns:
        torch.Tensor: (2, n_edges) int64 tensor of edges
    """

    return edges_cython(points, triangles)


def edges_cython(
    points: torch.Tensor, triangles: torch.Tensor
) -> Tuple[np.array, np.array]:
    """Interface to the cython function edges_cython_core"""

    points = points.cpu().numpy()
    # triangles are sorted
    triangles = triangles.sort(dim=0)[0]
    triangles = triangles.cpu().numpy()

    edges = edges_cython_core(points.astype(np.float64), triangles.astype(np.int64))

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
    triangles = torch.from_numpy(triangles.copy().T).long()

    points = torch.from_numpy(mesh.points).float()

    return points, triangles
