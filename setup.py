import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extension = Extension(
    name="fast_edge_extraction._cython",
    sources=[
        "fast_edge_extraction/_cython.pyx",
    ],
    include_dirs=[np.get_include(), "fast_edge_extraction"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    ext_modules = cythonize(extension)
)
