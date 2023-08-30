"""
This function defines the intended public API of the probability package of modules. The main functions
which are expected to be used are:

- crps

The following functions are used to support data preparation and normalisation:

- adjust_fcst_for_crps
- crps_cdf_brier_decomposition

"""

from scores.probability.crps_impl import (
    adjust_fcst_for_crps,
    crps_cdf,
    crps_cdf_brier_decomposition,
)
