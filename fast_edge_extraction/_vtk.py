import torch
import pyvista as pv


def edges(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Return the edges of the mesh

    Returns:
        edges (|E|x2 torch.Tensor): the edges of the mesh
    """

    assert triangles.shape[1] == 3

    faces = triangles.clone().cpu().numpy()
    points = points.cpu().numpy()
    mesh = pv.PolyData.from_regular_faces(points, faces)

    wireframe = mesh.extract_all_edges(use_all_points=True)
    edges = wireframe.lines.reshape(-1, 3)[:, 1:]
    return torch.from_numpy(edges)
