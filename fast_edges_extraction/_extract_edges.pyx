# distutils: language = c

cimport cython

from libc.math cimport sqrt
cimport numpy as cnp
cnp.import_array()
import numpy as np

INT_DTYPE = np.int64
ctypedef cnp.int64_t INT_DTYPE_t


def extract_edges(INT_DTYPE_t [:, :] triangles):

    assert triangles.shape[1] == 3, "Triangles must be a (n_triangles, 3) array"

    # if needed convert to contiguous arrays
    # this is important for reading the data in C++
    triangles = np.ascontiguousarray(triangles, dtype=INT_DTYPE)

    #Compute neighbors
    cdef INT_DTYPE_t i, j, k, l
    cdef INT_DTYPE_t n_triangles = triangles.shape[0]

    cdef INT_DTYPE_t [:, :] adjacent_triangles = -1 * np.ones((3 * n_triangles, 2), dtype=INT_DTYPE)
    cdef INT_DTYPE_t [:, :] adjacent_points = -1 * np.ones((3 * n_triangles, 2), dtype=INT_DTYPE)
    cdef INT_DTYPE_t [:] edge_degrees = np.zeros(3 * n_triangles, dtype=INT_DTYPE)

    # edges store the (repeated edges)
    cdef INT_DTYPE_t [:, :] edges = np.zeros((3 * n_triangles, 2), dtype=INT_DTYPE)

    for i in range(n_triangles):

        edges[3 * i, 0] = triangles[i, 0]
        edges[3 * i, 1] = triangles[i, 1]

        edges[3 * i + 1, 0] = triangles[i, 0]
        edges[3 * i + 1, 1] = triangles[i, 2]

        edges[3 * i + 2, 0] = triangles[i, 1]
        edges[3 * i + 2, 1] = triangles[i, 2]

    # Sort edges by lexicographic sort
    edges = np.sort(np.asarray(edges), axis=1)
    cdef INT_DTYPE_t[:] order = np.lexsort((np.asarray(edges[:, 1]), np.asarray(edges[:, 0])))

    # Remove duplicates
    cdef INT_DTYPE_t[:, :] short_edges = np.zeros((3 * n_triangles, 2), dtype=INT_DTYPE)
    cdef INT_DTYPE_t n_keep = 1

    cdef INT_DTYPE_t[:] triangles_ref = np.repeat(np.arange(n_triangles, dtype=INT_DTYPE), repeats=3)
    cdef INT_DTYPE_t[:] indice_ref = np.tile(np.array([2, 1, 0], dtype=INT_DTYPE), reps=n_triangles)

    short_edges[0, :] = edges[order[0]]
    adjacent_triangles[0, 0] = triangles_ref[order[0]]
    adjacent_points[0, 0] = indice_ref[order[0]]
    edge_degrees[0] = 1

    for i in range(1, 3 * n_triangles):
        if (edges[order[i], 0] != edges[order[i - 1], 0]) or (edges[order[i], 1] != edges[order[i - 1], 1]):
            short_edges[n_keep, :] = edges[order[i], :]
            adjacent_triangles[n_keep, 0] = triangles_ref[order[i]]
            adjacent_points[n_keep, 0] = indice_ref[order[i]]
            edge_degrees[n_keep] = 1
            n_keep += 1
        else:
            adjacent_triangles[n_keep - 1, 1] = triangles_ref[order[i]]
            adjacent_points[n_keep - 1, 1] = indice_ref[order[i]]
            edge_degrees[n_keep - 1] += 1

    return (
        np.asarray(short_edges)[:n_keep, :],
        np.asarray(edge_degrees)[:n_keep],
        np.asarray(adjacent_triangles)[:n_keep, :],
        np.asarray(adjacent_points)[:n_keep, :],

    )
