"""
The philosphy is to import the public API during the init phase rather than leaving it to the user
"""
# pylint: disable=E0603

import scores.categorical
import scores.continuous
import scores.functions
import scores.pandas
import scores.probability
import scores.sample_data
import scores.stats.statistical_tests  # noqa: F401

__version__ = "v0.5"

__all__ = [
    "scores.categorical",
    "scores.continuous",
    "scores.functions",
    "scores.pandas",
    "scores.probability",
    "scores.sample_data",
    "scores.stats.statistical_tests",
]
