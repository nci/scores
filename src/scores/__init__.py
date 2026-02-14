"""
The philosphy is to import the public API during the init phase rather than leaving it to the user
"""

# pylint: disable=E0603

import scores.categorical
import scores.continuous
import scores.emerging
import scores.functions
import scores.pandas
import scores.plotdata
import scores.probability
import scores.processing
import scores.sample_data
import scores.stats.statistical_tests  # noqa: F401

__version__ = "2.5.0"

__all__ = [
    "scores.categorical",
    "scores.contingency",
    "scores.continuous",
    "scores.emerging",
    "scores.functions",
    "scores.pandas",
    "scores.probability",
    "scores.processing",
    "scores.sample_data",
    "scores.stats.statistical_tests",
]
