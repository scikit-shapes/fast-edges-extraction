from typing import Tuple, Union

import numpy as np

from ._extract_edges import extract_edges as extract_edges_cython


def extract_edges(
    triangles: Union[list, np.ndarray],
    return_adjacency: bool = False,

) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract the edges from a set of triangles.

    With default parameters, this function returns the edges of the triangles
    as a 2D array of shape (n_edges, 2), where each row contains the indices of
    the two points that define the edge.

    If `return_adjacency` is set to `True`, the function also returns the
    degrees of each edge, and the indices of the triangles and points that are
    adjacent to each edge (limited to 2 triangles and 2 points per edge).

    Parameters
    ----------
    triangles : np.ndarray
        A 2D array of shape (n_triangles, 3) containing the indices of the
        points that define each triangle.

    return_adjacency : bool, optional
        If `True`, the function also returns the degrees of each edge, and the
        indices of the triangles and points that are adjacent to each edge
        (limited to 2 triangles and 2 points per edge). Default is `False`.

    Returns
    -------
    edges : np.ndarray
        A 2D array of shape (n_edges, 2) containing the indices of the points
        that define each edge.

    degrees : np.ndarray
        A 1D array of shape (n_edges,) containing the degree of each edge. This
        is the number of triangles that share the edge. Only returned if
        `return_adjacency` is `True`.

    adjacent_triangles : np.ndarray
        A 2D array of shape (n_edges, 2) containing the indices of the triangles
        that are adjacent to each edge. Only returned if `return_adjacency` is
        `True`.

    adjacent_points : np.ndarray
        A 2D array of shape (n_edges, 2) containing the indices of the points
        that are adjacent to each edge. Only returned if `return_adjacency` is
        `True`.
    """
    triangles = np.asarray(triangles)

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
