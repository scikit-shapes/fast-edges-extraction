"""Assert that the different implementations of edge extraction are consistent."""
from pyvista import examples
import numpy as np
import fast_edge_extraction
import torch


def test_consistency():
    mesh = examples.download_bunny().clean()

    points, triangles = fast_edge_extraction.compute_points_and_triangles(mesh)

    print(f"Shape of points: {points.shape}")
    print(f"Shape of triangles: {triangles.shape}")

    edges_torch = fast_edge_extraction.edges_torch(points, triangles)
    edges_cython = fast_edge_extraction.edges_cython(points, triangles)
    edges_vtk = fast_edge_extraction.edges_vtk(points, triangles)

    print(f"Shape of edges_torch: {edges_torch.shape}")
    print(f"Shape of edges_cython: {edges_cython.shape}")
    print(f"Shape of edges_vtk: {edges_vtk.shape}")

    edges_torch = fast_edge_extraction.sort_edges(edges_torch)
    edges_cython = fast_edge_extraction.sort_edges(edges_cython)
    edges_vtk = fast_edge_extraction.sort_edges(edges_vtk)

    assert np.allclose(edges_torch, edges_cython)
    assert np.allclose(edges_torch, edges_vtk)
