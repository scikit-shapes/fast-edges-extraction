fast-edge-extraction
====================

Extract the edges from the triangle structure of a triangle mesh.


Extract the edges from the triangles
------------------------------------

```python
import pyvista as pv
from pyvista import examples

bunny = examples.download_bunny()
triangles = bunny.bunny.regular_faces

from fast_edge_extraction import extract_edges

edges = extract_edges(triangles)
```

Also extract adjacency information
----------------------------------

```python
# Extract edges, degrees, and adjacency information (triangles and points)
edges, degrees, t_a, p_a = extract_edges(triangles, return_adjacency=True)

# Extract manifold (2 adjacent triangles) and boundary (1 adjacent triangle) edges
manifold_edges = edges[degrees == 2]
boundary_edges = edges[degrees == 1]
other = degrees > 2

print("Number of manifold edges:", manifold_edges.sum())
print("Number of boundary edges:", boundary_edges.sum())
print("Number of other edges:", other.sum())

# For each manifold edge, extract the adjacent triangles and points
manifold_adjacent_triangles = t_a[degrees == 2]
manifold_adjacent_points = p_a[degrees == 1]

# For each boundary edge, extract the adjacent triangle and point
boundary_adjacent_triangles = t_a[degrees == 1][:, 0]
boundary_adjacent_points = p_a[degrees == 1][:, 0]
```
