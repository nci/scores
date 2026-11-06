"""
This module contains the implementation of kernel-based probability scoring functions.
"""
from typing import Literal, NamedTuple

import numpy as np
import xarray as xr

from scores.typing import XarrayLike

Estimator = Literal["ecdf", "fair"]

class KernelTerms(NamedTuple):
    """
    Holds the terms required for kernel-based scorecomputations.
    """
    obs_term: xr.DataArray
    spread_sum: xr.DataArray
    n_members: xr.DataArray


def _spread_denominator(ens_count: xr.DataArray, method: Estimator)->XarrayLike:
    """Return :math: `2K`, which is the constant that normalises the kernel spread term."""
    if method == "fair":
        return 2 * ens_count * (ens_count - 1)
    if method == "ecdf":
        return 2 * ens_count**2
    else:
        raise ValueError(f"Unsupported kernel method: {method}")

def _spread_term(terms:KernelTerms, method: Estimator)->XarrayLike:
    """ Return the normalised spread term, :math:`\\frac{1}{2K}\\sum_i\\sum_j \\rho(v(x_i), v(x_j))`"""
    return terms.spread_sum / _spread_denominator(terms.n_members, method)

def _combine_kernel_terms(terms: KernelTerms, method: Estimator) -> XarrayLike:
    """Combine the observation and spread terms to compute the final kernel-based score."""
    return terms.obs_term - _spread_term(terms, method)

def _check_kernel_method(method: str):
    if method not in ["ecdf", "fair"]:
        raise ValueError("`method` must be one of 'ecdf' or 'fair'")

def _crps_engine(fcst: XarrayLike, obs: XarrayLike, ensemble_member_dim: str):
    """
    Computes the CRPS score using a kernel-based approach.

        Here we implement a faster method to compute the ensemble spread term
    and explain the derivation below.
    We want to compute:
        S = sum_{i=1}^m sum_{j=1}^m |x_i - x_j|

    The naive approach requires O(m²) time, but by first sorting the ensemble
    members x in ascending order, the computation can be reduced to O(m log m).

    Derivation:
      Start by separating the double sum into the i < j and i > j parts:
          S = sum_{i<j} |x_i - x_j| + sum_{i>j} |x_i - x_j|
            (the i = j terms are zero)

      Since the array is sorted, we have x_j ≥ x_i when i < j,
      The sum_{i>j} |x_i - x_j| term is identical to the sum_{i<j} term, so:
          S = 2 sum_{i<j} (x_j - x_i)

      Each x_j appears (j − 1) times as a positive term,
      and each x_i appears (m − i) times as a negative term:
          S = 2 [ sum_{j=1}^m (j − 1) x_j  −  sum_{i=1}^m (m − i) x_i ]

      Combine and simplify coefficients (using 'i' as the index for both):
          S = 2 sum_{i=1}^m (2i − m − 1)  x_i

    where x_(i) are the sorted ensemble members (ranked from smallest to largest).

    The idea of sorting the ensemble members to compute the CRPS spread term
    can be found in Hersbach, H. (2000). Decomposition of the Continuous Ranked Probability
    Score for Ensemble Prediction Systems. Weather and Forecasting, 15(5), 559–570.
    https://doi.org/10.1175/1520-0434(2000)015<0559:dotcrp>2.0.co;2
    """
    # Calculate forecast spread term
    ens_count = fcst.count(ensemble_member_dim)  # Get number of non-NaN members for each point
    m_total = fcst.sizes[ensemble_member_dim]  # Get m_total (total dimension size)

    # Sort forecast values along the member dim.
    fcst_sorted = xr.apply_ufunc(
        np.sort,
        fcst,
        input_core_dims=[[ensemble_member_dim]],
        output_core_dims=[[ensemble_member_dim]],
        exclude_dims=set((ensemble_member_dim,)),
        kwargs={"axis": -1},
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {ensemble_member_dim: m_total}},
    )

    i = xr.DataArray(np.arange(m_total), dims=[ensemble_member_dim])
    coeffs = 2 * i - ens_count + 1
    fcst_spread_term_numerator = 2 * (fcst_sorted * coeffs).sum(dim=ensemble_member_dim, skipna=True)

    obs_term = abs(fcst - obs).mean(dim=ensemble_member_dim)

    return KernelTerms(obs_term=obs_term, spread_sum=fcst_spread_term_numerator, n_members=ens_count)

