"""
Neutrino Mass Matrix Research Package

This package provides tools for analyzing neutrino mass matrices
in the context of Seesaw and Inverse Seesaw mechanisms.

Modules:
- matrix_utils: General matrix operations and diagonalization utilities
- seesaw: Implementation of various Seesaw mechanisms (Type I, II, III)
- inverse_seesaw: Implementation of Inverse Seesaw mechanism
- pmns_matrix: PMNS mixing matrix parameterizations and calculations
- mass_ordering: Tools for analyzing normal and inverted mass hierarchies
- visualization: Plotting and visualization utilities for mass spectra and mixing
"""

__version__ = "0.1.0"
__author__ = "Neutrino Physics Research"

from .matrix_utils import *
from .seesaw import *
from .inverse_seesaw import *
from .pmns_matrix import *
