from numba import jit
import numpy as np
import torch
import pyvista

@jit(nopython=True, fastmath=True)
def sort_edges(edges, inverse_ordering):
    """Sort the edges of a mesh, given inverse ordering
    Args:
        edges (array): the edges of the mesh
        inverse_ordering (list): the inverse order
    Returns:
        sorted_edges (array): the sorted edges
    """
    sorted_edges = np.repeat(2, len(edges))
    for i in range(len(edges)):
        if i % 3 != 0:
            sorted_edges[i] = inverse_ordering[edges[i]]

    return sorted_edges
    


def edges(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Return the edges of the mesh

    Returns:
        edges (2x|E| torch.Tensor): the edges of the mesh
    """

    # Back to pyvista padded faces format
    triangles_pad = triangles.clone().cpu().numpy().T
    triangles_pad = np.pad(triangles_pad, ((0, 0), (1, 0)), mode='constant', constant_values=3).reshape(-1)

    points_np = points.cpu().numpy()

    shape = pyvista.PolyData(points_np, faces=triangles_pad)

    edges_mesh = shape.extract_all_edges()
    edges_ordering = np.lexsort((edges_mesh.points[:,2], edges_mesh.points[:,1], edges_mesh.points[:,0]))
    inverse_edges_ordering = np.argsort(edges_ordering)
    
    points_ordering = np.lexsort((edges_mesh.points[:,2], edges_mesh.points[:,1], edges_mesh.points[:,0]))
    

    edges = sort_edges(edges_mesh.lines, inverse_edges_ordering) # To lexicographic order
    edges = sort_edges(edges, points_ordering) # Back to the original order
    edges = edges.reshape(-1, 3)[:, 1:] # Remove padding
    edges = torch.Tensor(edges).T.long()
    return edges