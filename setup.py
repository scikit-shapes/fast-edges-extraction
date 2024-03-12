import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extension = Extension(
    name="fast_edges_extraction._extract_edges",
    sources=[
        "fast_edges_extraction/_extract_edges.pyx",
    ],
    include_dirs=[np.get_include(), "fast_edges_extraction"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    ext_modules = cythonize(extension)
)
