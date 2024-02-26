"""Assert that the different implementations of edge extraction are consistent."""

import numpy as np

import fast_edge_extraction

# VTK and np implementations


def extract_edges_numpy(
    points: np.ndarray, triangles: np.ndarray  # noqa: ARG001
) -> np.ndarray:
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

    edges_np = extract_edges_numpy(points, triangles)
    edges_cython = fast_edge_extraction.extract_edges(points, triangles)

    print(f"Shape of edges_np: {edges_np.shape}")
    print(f"Shape of edges_cython: {edges_cython.shape}")

    edges_np = fast_edge_extraction.sort_edges(edges_np)
    edges_cython = fast_edge_extraction.sort_edges(edges_cython)

    assert np.allclose(edges_np, edges_cython)
