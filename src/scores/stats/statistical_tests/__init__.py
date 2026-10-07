"""
Import the functions from the implementations into the public API
"""

from scores.stats.statistical_tests.diebold_mariano_impl import (
    diebold_mariano,
    diebold_mariano_1d,
)

__all__ = ["diebold_mariano", "diebold_mariano_1d"]
