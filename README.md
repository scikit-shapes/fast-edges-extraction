# fast-edge-extraction
Fast edge extraction for triangle mesh

This repository was create to find the fastest solution to the following problem : given an array of integer repredenting a collection of triangles, find the associated list of edges. VTK provides a functionality to extract edges from a triangle mesh, but the output is a new mesh where points are not in the same order than in the original mesh.

Compared implementations are :
* VTK + make indices in correspondance with original mesh (with Numba)
* Pytorch extraction of edges + cleaning to avoid repeating edges
* Cython implementation

Tests are made to ensure consistency accross implementation.

So far, the cython implementation is the fastest. Decision were made to integrate it in scikit-shapes.
