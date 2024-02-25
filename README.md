# fast-edge-extraction

TODO:

- Propper package sctructure: repo-review
- Basic documentation
- Compilation + PyPI publication
- conda publication

Fast edge extraction for triangle mesh

This repository was created to find the fastest solution to the following problem : given a (n_triangle, 3) array of triangles with shape , find the associated (n_edges, 2) array of  of edges.

Compared implementations are :
* VTK (through pyvista, with `use_all_points=True` to keep correspondence with initial points)
* Pytorch extraction of edges + cleaning to avoid repeating edges
* Cython implementation

Tests are made to ensure consistency accross implementation.

A benchmark of run times can be found at `examples/compare_running_times.py`

```
210873 points, 421965 triangles
-----------------------------------
|Implementation    | Running time |
|------------------+---------------
|VTK               | 0.312        |
|Torch             | 3.727        |
|Cython            | 0.281        |
----------------------------------
Number of edges: 633083
```

Cython and VTK have similar running times while PyTorch implementation is slower. Cython implementation give access to a more explicit description of the edge topology : degrees, adjacent points, without any overhead.

Decision were made to integrate the cython implementation in scikit-shapes.
