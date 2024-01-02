import torch


def edges(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Return the edges of the mesh

    Args:
        points (torch.Tensor): the points of the mesh
        triangles (torch.Tensor): the triangles of the mesh

    Returns:
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
