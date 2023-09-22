from setuptools import setup, Extension, find_packages
import numpy as np
from Cython.Build import cythonize

extension = Extension(
    name="fast_edge_extraction._cython",
    sources=[
        "fast_edge_extraction/_cython.pyx",
    ],
    include_dirs=[np.get_include(), "fast_edge_extraction"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)


dependencies = [
    "pyvista",
    "numba",
    "numpy",
    "torch",
]


setup(
    name="fast-edge-extraction",
    version="0.1",
    description="Fast edges extraction from a triangle mesh",
    author="Louis Pujol",
    url="",
    setup_requires=["cython"] + dependencies,
    install_requires=dependencies,
    packages=find_packages(),
    package_data={
        "fast_edge_extraction": ["*.pyx", "*.pxd"],
    },
    ext_modules=cythonize(extension),
)
