from setuptools import setup


dependencies = [
    "pyvista",
    "numba",
    "numpy",
    "torch",
]


setup(
    name="fast-edges-extraction",
    version="0.1",
    description="Fast edges extraction from a triangle mesh",
    author="Louis Pujol",
    url="",
    install_requires=dependencies,
    packages=["fast_edge_extraction"],
)
