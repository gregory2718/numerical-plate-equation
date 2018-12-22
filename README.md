# numerical-plate-equation

This repository contains a set of files for solving the plate equation across specified domains, using both finite element methods with FEniCS and discrete exterior calculus.

# Files

**mesh_gen.py** contains code for creating meshes using <a href="https://pypi.org/project/pygmsh/">pygmsh</a>. This code expects a copy of <a href="https://people.sc.fsu.edu/~jburkardt/py_src/dolfin-convert/dolfin-convert.html">dolfin-convert</a> (renamed to dolfin_convert.py) to exist in its local directory.

**plate_dec.py** contains `get_eigs_PyDEC(...)`, which takes a mesh generated with mesh_gen.py and computes a set of eigenvector/value pairs of the plate equation for it using <a href="https://github.com/hirani/pydec">PyDEC</a>.

**plate_fenics.py** contains `get_eigs_FEniCS(...)`, which is the same as `get_eigs_PyDEC(...)` except the solutions are computed via <a href="https://fenicsproject.org/">FEniCS</a>.

**processing_data.py** contains miscellaneous code for processing the solutions found by plate_dec.py and plate_fenics.py. This includes comparing the computed eigenvalue spectrums to analytically derived equivalents, rendering eigenstates, and computing the error in the computed eigenvectors with respect to given truth data. Generally truth data comes from high density meshes, but for simple domains it can also be analytically derived.
