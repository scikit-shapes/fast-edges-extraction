from typing import Tuple

import numpy as np

from ._cython import edges as edges_cython_core


def extract_edges(
    points: np.ndarray, triangles: np.ndarray # noqa: ARG001
) -> Tuple[np.ndarray, np.ndarray]:
    """Interface to the cython function edges_cython_core"""
    if triangles.shape[1] != 3:
        msg = "Triangles should have shape (n_triangles, 3)"
        raise ValueError(msg)
    return edges_cython_core(triangles.astype(np.int64))[0]


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
