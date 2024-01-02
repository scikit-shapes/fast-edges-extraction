import torch
import pyvista as pv


def edges(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Return the edges of the mesh

    Returns:
        edges (2x|E| torch.Tensor): the edges of the mesh
    """

    faces = triangles.clone().cpu().numpy()
    points = points.cpu().numpy()
    mesh = pv.PolyData.from_regular_faces(points, faces)

    wireframe = mesh.extract_all_edges(use_all_points=True)
    edges = wireframe.lines.reshape(-1, 3)[:, 1:]
    return torch.from_numpy(edges)
