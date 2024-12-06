"""
Import the functions from the implementations into the public API
"""

from scores.continuous.correlation.correlation_impl import pearsonr, spearmanr

__all__ = ["pearsonr", "spearmanr"]
