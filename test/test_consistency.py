"""Assert that the different implementations of edge extraction are consistent."""
from pyvista import examples
import numpy as np
import fast_edge_extraction
import torch


def normalize_edges(edges: torch.Tensor) -> np.array:
    """Lexicographically sort a (2, n_edges) array of edges to
    allow for comparison with various implementations of edges
    extraction.

    Args:
        edges (torch.Tensor): a (2, n_edges) array of edges

    Returns:
        np.array: the sorted edges
    """
    edges = edges.sort(dim=0)[0].cpu().numpy()
    ordering = np.lexsort((edges[1], edges[0]))
    return edges[:, ordering]


def test_consistency():
    mesh = examples.download_bunny().clean()

    points, triangles = fast_edge_extraction.compute_points_and_triangles(mesh)

    edges_numba = fast_edge_extraction.edges_numba(points, triangles)
    edges_torch = fast_edge_extraction.edges_torch(points, triangles)
    edges_cython = fast_edge_extraction.edges_cython(points, triangles)

    edges_numba = normalize_edges(edges_numba)
    edges_torch = normalize_edges(edges_torch)
    edges_cython = normalize_edges(edges_cython)

    assert np.allclose(edges_numba, edges_torch)
    assert np.allclose(edges_numba, edges_cython)
