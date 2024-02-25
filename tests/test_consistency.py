"""Assert that the different implementations of edge extraction are consistent."""
from pyvista import examples
import pyvista as pv
import numpy as np
import fast_edge_extraction
import torch

def extract_edges_torch(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
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


def extract_edges_vtk(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Return the edges of the mesh

    Parameters
    ----------
    points
        the points of the mesh
    triangles
        the triangles of the mesh

    Returns
    -------
        edges (|E|x2 torch.Tensor): the edges of the mesh
    """

    assert triangles.shape[1] == 3

    faces = triangles.clone().cpu().numpy()
    points = points.cpu().numpy()
    mesh = pv.PolyData.from_regular_faces(points, faces)

    wireframe = mesh.extract_all_edges(use_all_points=True)
    edges = wireframe.lines.reshape(-1, 3)[:, 1:]
    return torch.from_numpy(edges)


def test_consistency():
    mesh = examples.download_bunny().clean()

    points, triangles = fast_edge_extraction.compute_points_and_triangles(mesh)

    print(f"Shape of points: {points.shape}")
    print(f"Shape of triangles: {triangles.shape}")

    edges_torch = extract_edges_torch(points, triangles)
    edges_cython = fast_edge_extraction.extract_edges(points, triangles)
    edges_vtk = extract_edges_vtk(points, triangles)

    print(f"Shape of edges_torch: {edges_torch.shape}")
    print(f"Shape of edges_cython: {edges_cython.shape}")
    print(f"Shape of edges_vtk: {edges_vtk.shape}")

    edges_torch = fast_edge_extraction.sort_edges(edges_torch)
    edges_cython = fast_edge_extraction.sort_edges(edges_cython)
    edges_vtk = fast_edge_extraction.sort_edges(edges_vtk)

    assert np.allclose(edges_torch, edges_cython)
    assert np.allclose(edges_torch, edges_vtk)
