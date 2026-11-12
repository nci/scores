"""
Private helper functions for kernel-based ensemble scores, such as the CRPS for ensemble forecasts.

Kernel scores are computed here in the form

.. math::
    S(F, y) = \\mathbb{E}\\,\\rho\\big(v(X), v(y)\\big) - \\tfrac{1}{2}\\,\\mathbb{E}\\,\\rho\\big(v(X), v(X')\\big)

where

- :math:`F` is the forecast distribution and :math:`y` is the observation.
- :math:`X` and :math:`X'` are independent random draws from :math:`F`. For an ensemble,
  these are the ensemble members.
- :math:`v` maps forecasts and observations into the space where they are compared. For the
  CRPS it is the identity, or a chaining function for the threshold-weighted CRPS.
- :math:`\\rho` is the kernel, which measures how far apart two values are and must satisfy
  :math:`\\rho(a, a) = 0`. For the CRPS, :math:`\\rho(a, b) = |a - b|`.
"""

from typing import NamedTuple, get_args

import numpy as np
import xarray as xr

from scores.typing import Estimator, XarrayLike


class KernelTerms(NamedTuple):
    """
    Terms required to compute a kernel-based score for an ensemble forecast.

    Attributes:
        obs_term: Mean over the non-NaN members of :math:`\\rho(v(x_i), v(y))`.
        spread_sum: Sum of :math:`\\rho(v(x_i), v(x_j))` (the kernel function) over all
            ordered non-NaN pairs of ensemble members, before dividing by :math:`2K`,
            where :math:`K = m^2` for the "ecdf" estimator and :math:`K = m(m - 1)`
            for the "fair" estimator.
        n_members: Number of non-NaN ensemble members, :math:`m`.
    """

    obs_term: XarrayLike
    spread_sum: XarrayLike
    n_members: XarrayLike


def _spread_denominator(n_members: XarrayLike, method: Estimator) -> XarrayLike:
    """
    Calculates the denominator that normalises the kernel spread sum.

    Args:
        n_members: Number of ensemble members, :math:`m`.
        method: Estimator to use, either "ecdf" or "fair".

    Returns:
        :math:`2m(m-1)` for the "fair" estimator, or :math:`2m^2` for the "ecdf" estimator,
        where :math:`m` is ``n_members``. Non-positive values are returned as NaN, which
        occurs when there are no members, or one member with the "fair" estimator.

    Raises:
        ValueError: If ``method`` is not one of the supported estimators.
    """
    if method == "fair":
        denominator = 2 * n_members * (n_members - 1)
    elif method == "ecdf":
        denominator = 2 * n_members**2
    else:
        raise ValueError(f"Unknown method: {method}")
    return denominator.where(denominator > 0)


def _spread_term(terms: KernelTerms, method: Estimator) -> XarrayLike:
    """
    Returns the normalised spread term,
    :math:`\\frac{1}{2K}\\sum_{i=1}^{m}\\sum_{j=1}^{m} \\rho\\big(v(x_i), v(x_j)\\big)`,
    where :math:`2K` is the normalising denominator returned by ``_spread_denominator``.

    Args:
        terms: Kernel terms holding the spread sum and number of ensemble members.
        method: Estimator to use, either "ecdf" or "fair".

    Returns:
        The spread sum divided by :math:`2K`, as given by ``_spread_denominator``.
    """
    return terms.spread_sum / _spread_denominator(terms.n_members, method)


def _combine_kernel_terms(terms: KernelTerms, method: Estimator) -> XarrayLike:
    """
    Combine the observation and spread terms to compute the kernel-based score.

    Args:
        terms: Kernel terms for the forecast and observation.
        method: Estimator to use, either "ecdf" or "fair".

    Returns:
        The observation term minus the normalised spread term.
    """
    return terms.obs_term - _spread_term(terms, method)


def _check_estimator(method: str) -> None:
    """
    Check that `method` is one of the supported estimators in ``Estimator``.

    Args:
        method: Estimator name to check.

    Raises:
        ValueError: If `method` is not "ecdf" or "fair".
    """
    valid = get_args(Estimator)
    if method not in valid:
        options = " or ".join(repr(estimator) for estimator in valid)
        raise ValueError(f"`method` must be one of {options}")


def _crps_terms(fcst: XarrayLike, obs: XarrayLike, ensemble_member_dim: str) -> KernelTerms:
    """
    Computes the terms of the kernel-based CRPS for an ensemble forecast, where the
    kernel is :math:`\\rho(a, b) = |a - b|`.

    Args:
        fcst: Ensemble forecast, with members along ``ensemble_member_dim``. NaN members are ignored.
        obs: Observations, broadcastable against ``fcst`` along its non-member dimensions.
        ensemble_member_dim: Name of the ensemble member dimension in ``fcst``.

    Returns:
        The ``KernelTerms`` (observation term, spread sum and member count) used by the
        estimator-specific functions.

    Notes:
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
    n_members = fcst.count(ensemble_member_dim)  # Get number of non-NaN members for each point
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
    coeffs = 2 * i - n_members + 1
    spread_sum = 2 * (fcst_sorted * coeffs).sum(dim=ensemble_member_dim, skipna=True)

    obs_term = abs(fcst - obs).mean(dim=ensemble_member_dim)

    return KernelTerms(obs_term=obs_term, spread_sum=spread_sum, n_members=n_members)
