
from .numba import edges as edges_numba
from .torch import edges as edges_torch
from typing import Tuple

import pyvista
import torch

def compute_points_and_triangles(mesh: pyvista.PolyData) -> Tuple[torch.Tensor, torch.Tensor]:
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