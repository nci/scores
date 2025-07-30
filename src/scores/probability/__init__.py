"""
Import the functions from the implementations into the public API
"""

from scores.plotdata.murphy_impl import murphy_score, murphy_thetas
from scores.plotdata.roc_impl import roc as roc_curve_data
from scores.probability.brier_impl import brier_score, brier_score_for_ensemble
from scores.probability.crps_impl import (
    adjust_fcst_for_crps,
    crps_cdf,
    crps_cdf_brier_decomposition,
    crps_for_ensemble,
    crps_step_threshold_weight,
    interval_tw_crps_for_ensemble,
    tail_tw_crps_for_ensemble,
    tw_crps_for_ensemble,
)
from scores.processing.isoreg_impl import isotonic_fit

if roc_curve_data.__doc__ is not None:
    roc_curve_data.__doc__ = roc_curve_data.__doc__.replace(
        "from scores.plotdata import roc", "from scores.probablity import roc_curve_data"
    )

__all__ = [
    "murphy_score",
    "murphy_thetas",
    "brier_score",
    "brier_score_for_ensemble",
    "adjust_fcst_for_crps",
    "crps_cdf",
    "crps_cdf_brier_decomposition",
    "crps_for_ensemble",
    "roc_curve_data",
    "isotonic_fit",
    "crps_step_threshold_weight",
    "tw_crps_for_ensemble",
    "tail_tw_crps_for_ensemble",
    "interval_tw_crps_for_ensemble",
]
