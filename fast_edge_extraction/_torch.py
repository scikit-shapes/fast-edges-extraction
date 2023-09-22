import torch


def edges(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Return the edges of the mesh

    Args:
        points (torch.Tensor): the points of the mesh
        triangles (torch.Tensor): the triangles of the mesh

    Returns:
        edges (2x|E| torch.Tensor): the edges of the mesh
    """
    # Compute the edges of the triangles and sort them
    repeated_edges = torch.concat(
        [
            triangles[[0, 1], :],
            triangles[[1, 2], :],
            triangles[[0, 2], :],
        ],
        dim=1,
    ).sort(dim=0)[0]
    edges = torch.unique(repeated_edges, dim=1)
    return edges
