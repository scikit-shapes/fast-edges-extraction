from typing import Tuple, Union

import numpy as np

from ._extract_edges import extract_edges as extract_edges_cython


def extract_edges(
    triangles: np.ndarray,
    return_adjacency: bool = False,

) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Interface to the cython function edges_cython_core"""

    if triangles.shape[1] != 3:
        msg = "Triangles should have shape (n_triangles, 3)"
        raise ValueError(msg)

    (
        edges,
        degrees,
        adjacent_triangles,
        adjacent_points
    ) = extract_edges_cython(triangles.astype(np.int_))

    if not return_adjacency:
        return edges

    return edges, degrees, adjacent_triangles, adjacent_points
