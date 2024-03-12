"""Assert that the different implementations of edge extraction are consistent."""

import numpy as np

import fast_edges_extraction

# VTK and np implementations


def sort_edges(edges: np.ndarray) -> np.ndarray:
    """Lexicographically sort a (n_edges, 2) array of edges to
    allow for comparison with various implementations of edges
    extraction.

    Args:
        edges (torch.Tensor): a (n_edges, 2) array of edges

    Returns:
        np.array: the sorted edges
    """
    assert edges.shape[1] == 2
    edges.sort(axis=1)
    ordering = np.lexsort((edges[:, 1], edges[:, 0]))
    return edges[ordering]


def extract_edges_numpy(triangles: np.ndarray) -> np.ndarray:
    """Return the edges of the mesh

    Parameters
    ----------
    points
        the points of the mesh
    triangle
        the triangles of the mesh

    Returns
    -------
        edges (|E|x2 np.ndarray): the edges of the mesh
    """
    assert triangles.shape[1] == 3
    # Compute the edges of the triangles and sort them
    repeated_edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [0, 2]],
        ],
        axis=0,
    )
    repeated_edges.sort(axis=1)
    return np.unique(repeated_edges, axis=0)


def test_consistency():
    generator = np.random.default_rng(0)
    points = generator.random((100, 3))
    triangles = generator.integers(0, 500, (500, 3))

    print(f"Shape of points: {points.shape}")
    print(f"Shape of triangles: {triangles.shape}")

    edges_np = extract_edges_numpy(triangles)
    edges_cython = fast_edges_extraction.extract_edges(triangles)

    print(f"Shape of edges_np: {edges_np.shape}")
    print(f"Shape of edges_cython: {edges_cython.shape}")

    edges_np = sort_edges(edges_np)
    edges_cython = sort_edges(edges_cython)

    assert np.allclose(edges_np, edges_cython)
