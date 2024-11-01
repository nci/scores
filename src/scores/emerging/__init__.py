"""
Import the functions from the implementations into the public API.

This module is intended to hold code for new metrics that may not yet have a publication
and may not yet be in their final form, but which still have interest and value to
the verification community.

To propose a metric for inclusion in this API, please either get in touch through the
"Discussions" area of the github page (https://github.com/nci/scores/discussions) or
raise a new issue (https://github.com/nci/scores/issues)
"""

from scores.emerging.emerging_impl import risk_matrix_score, matrix_weights_to_array, scaling_to_weight_array

__all__ = [
    "risk_matrix_score",
    "matrix_weights_to_array",
    "scaling_to_weight_array",
]
