"""
This module contains functions to calculate the correlation.
"""

from typing import Optional

import xarray as xr

import scores.utils
from scores.typing import FlexibleDimensionTypes, XarrayLike, all_same_xarraylike, is_xarraylike
from scores.utils import check_weights


def anomaly_correlation_coefficient(
    fcst: XarrayLike,
    obs: XarrayLike,
    climatology: XarrayLike,
    *,
    centered: bool = False,
    reduce_dims: FlexibleDimensionTypes | None = None,
    preserve_dims: FlexibleDimensionTypes | None = None,
    weights: XarrayLike | None = None,
) -> XarrayLike:
    """Calculate the anomaly correlation coefficient (ACC).

    ACC measures the similarity between forecast and observed anomalies, where
    each anomaly is calculated relative to the same climatology. This function
    defaults to uncentred ACC. Set ``centered=True`` for centred ACC.

    The uncentred form is

    .. math::

        \\operatorname{ACC} =
        \\frac{\\sum_{i \\in D} w_i (x_i - c_i)(y_i - c_i)}
        {\\sqrt{
            \\sum_{i \\in D} w_i (x_i - c_i)^2
            \\sum_{i \\in D} w_i (y_i - c_i)^2
        }}

    where :math:`x_i` is the forecast, :math:`y_i` is the observation,
    :math:`c_i` is the climatology, :math:`w_i` is the optional weight, and
    :math:`D` is the set of points along the reduced dimensions. With no
    weights, :math:`w_i = 1`.

    For centred ACC, replace each anomaly :math:`a_i = x_i - c_i` and
    :math:`b_i = y_i - c_i` in the formula above by :math:`a_i - \\bar{a}_w`
    and :math:`b_i - \\bar{b}_w`, respectively, where
    :math:`\\bar{a}_w = \\sum_{i \\in D} w_i a_i / \\sum_{i \\in D} w_i`
    and similarly for :math:`\\bar{b}_w`. These means use the same valid
    pairs and reduction dimensions as the correlation. Centring removes
    uniform offsets in either anomaly field over those dimensions; uncentred
    ACC retains the mean anomalies and can change with such offsets.

    The two forms are discussed in Chapter 8 of Wilks (2011).

    ACC ranges from -1 to 1 when it is defined. A value of 1 indicates that the
    anomaly fields point in the same direction, 0 indicates no projection of
    one anomaly field onto the other, and -1 indicates opposing anomaly
    fields. ACC evaluates pattern agreement; a high value does not by itself
    imply small errors or unbiased forecasts. The result is ``NaN`` wherever
    either anomaly field has zero weighted magnitude (after centring, if
    requested). In particular, centred ACC is undefined if either anomaly
    field is constant over the positively weighted valid points.

    Inputs are aligned on their shared coordinates. Missing values are handled
    using pairwise-complete samples. If ``fcst``,
    ``obs``, or ``climatology`` is missing at a point, that point is omitted
    from every sum and mean. The result is ``NaN`` if no valid, positively weighted
    points remain in a reduced group.

    Args:
        fcst: Forecast or predicted values.
        obs: Observed values.
        climatology: Climatological reference values subtracted from both
            ``fcst`` and ``obs``. Dimensions omitted from ``climatology`` are
            broadcast by xarray.
        centered: If ``True``, subtract the weighted sample mean of each
            anomaly field before calculating ACC. Defaults to ``False``
            (uncentred ACC).
        reduce_dims: Dimensions to reduce when calculating ACC. All other
            dimensions are preserved. By default, all dimensions are reduced.
        preserve_dims: Dimensions to preserve when calculating ACC. All other
            dimensions are reduced. This argument is mutually exclusive with
            ``reduce_dims``. Preserving all dimensions is not supported because
            ACC requires at least one dimension of every data variable to be
            reduced.
        weights: Non-negative weights applied to the reduced points, such as
            grid-cell area weights. Weights must be broadcastable to the input
            data, contain no missing values, and contain at least one positive
            value. A ``DataArray`` can weight either ``DataArray`` or
            ``Dataset`` inputs. Dataset weights must contain the same data
            variables as the other Dataset inputs.

    Returns:
        An xarray object containing ACC values for each combination of the
        preserved dimensions. The return type matches the forecast type.

    Raises:
        TypeError: If ``fcst``, ``obs``, and ``climatology`` are not all
            ``DataArray`` objects or all ``Dataset`` objects, if ``weights`` is
            not an xarray object, or if Dataset weights are supplied for
            DataArray inputs.
        ValueError: If Dataset inputs do not contain the same data variables,
            if no dimensions are reduced for one or more data variables, if
            dimension arguments are invalid, or if weights are negative,
            contain missing values, or do not contain a positive value.

    References:
        - WWRP/WGNE Joint Working Group on Forecast Verification Research.
          (n.d.). *Forecast verification: Methods, issues and FAQ*.
          https://jwgfvr.github.io/forecastverification/
        - Jolliffe, I. T., & Stephenson, D. B. (Eds.). (2012).
          *Forecast verification: A practitioner's guide in atmospheric
          science* (2nd ed.). Wiley. https://doi.org/10.1002/9781119960003
        - Wilks, D. S. (2011). *Statistical methods in the atmospheric
          sciences* (3rd ed.). Academic Press.

    See Also:
        :py:func:`scores.continuous.correlation.pearsonr`

    Examples:
        >>> import xarray as xr
        >>> from scores.continuous.correlation import (
        ...     anomaly_correlation_coefficient,
        ... )

        >>> climatology = xr.DataArray(
        ...     [10.0, 10.0, 10.0],
        ...     dims="location",
        ...     coords={"location": ["A", "B", "C"]},
        ... )
        >>> fcst = xr.DataArray(
        ...     [[12.0, 11.0, 9.0], [11.0, 10.0, 7.0]],
        ...     dims=["time", "location"],
        ... )
        >>> obs = xr.DataArray(
        ...     [[11.0, 12.0, 9.0], [12.0, 10.0, 8.0]],
        ...     dims=["time", "location"],
        ... )
        >>> anomaly_correlation_coefficient(
        ...     fcst, obs, climatology, reduce_dims="location"
        ... )
        <xarray.DataArray (time: 2)> Size: 16B
        array([0.83333333, 0.89442719])
        Dimensions without coordinates: time
    """
    if not all_same_xarraylike([fcst, obs, climatology]):
        raise TypeError("fcst, obs, and climatology must all be xarray DataArrays or all be xarray Datasets.")

    if weights is not None and not is_xarraylike(weights):
        raise TypeError("weights must be an xarray DataArray or xarray Dataset.")

    if isinstance(fcst, xr.DataArray) and isinstance(weights, xr.Dataset):
        raise TypeError("weights cannot be an xarray Dataset when the other inputs are xarray DataArrays.")

    if isinstance(fcst, xr.Dataset):
        data_vars = set(fcst.data_vars)
        if set(obs.data_vars) != data_vars or set(climatology.data_vars) != data_vars:
            raise ValueError("fcst, obs, and climatology Datasets must contain the same variables.")
        if isinstance(weights, xr.Dataset) and set(weights.data_vars) != data_vars:
            raise ValueError("Dataset weights must contain the same variables as the other inputs.")

    if weights is not None:
        check_weights(weights)

    # Use weights_dims to include broadcast dimensions from both climatology and weights.
    extra_dims = set(climatology.dims)
    if weights is not None:
        extra_dims.update(weights.dims)
    dims_to_reduce = scores.utils.gather_dimensions(
        fcst.dims,
        obs.dims,
        weights_dims=extra_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
    if not dims_to_reduce:
        raise ValueError("You cannot preserve all dimensions with anomaly_correlation_coefficient.")

    fcst_anomaly = fcst - climatology
    obs_anomaly = obs - climatology
    valid = fcst_anomaly.notnull() & obs_anomaly.notnull()
    if weights is not None:
        # Include the weights' coordinates and broadcast dimensions before centring.
        valid = valid & weights.notnull()
    fcst_anomaly = fcst_anomaly.where(valid)
    obs_anomaly = obs_anomaly.where(valid)

    # Dataset reductions ignore dimensions absent from a variable. Check that
    # each variable still has at least one dimension to reduce after broadcasting.
    if isinstance(valid, xr.Dataset) and any(
        dims_to_reduce.isdisjoint(variable.dims) for variable in valid.data_vars.values()
    ):
        raise ValueError("At least one dimension must be reduced for every data variable when calculating ACC.")

    if centered:
        sample_weights = valid if weights is None else valid * weights
        weight_sum = sample_weights.sum(dim=dims_to_reduce)
        weight_sum = weight_sum.where(weight_sum > 0)
        fcst_anomaly = fcst_anomaly - (fcst_anomaly * sample_weights).sum(dim=dims_to_reduce) / weight_sum
        obs_anomaly = obs_anomaly - (obs_anomaly * sample_weights).sum(dim=dims_to_reduce) / weight_sum

    cross_product = fcst_anomaly * obs_anomaly
    fcst_squared = fcst_anomaly**2
    obs_squared = obs_anomaly**2

    if weights is not None:
        cross_product = cross_product * weights
        fcst_squared = fcst_squared * weights
        obs_squared = obs_squared * weights

    numerator = cross_product.sum(dim=dims_to_reduce, skipna=True)
    fcst_magnitude = fcst_squared.sum(dim=dims_to_reduce, skipna=True) ** 0.5
    obs_magnitude = obs_squared.sum(dim=dims_to_reduce, skipna=True) ** 0.5
    denominator = fcst_magnitude * obs_magnitude

    return numerator / denominator.where(denominator > 0)


def pearsonr(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
) -> XarrayLike:
    """
    Calculates the Pearson's correlation coefficient between two xarray DataArrays

    .. math::
        \\rho = \\frac{\\sum_{i=1}^{n}{(x_i - \\bar{x})(y_i - \\bar{y})}}
        {\\sqrt{\\sum_{i=1}^{n}{(x_i-\\bar{x})^2}\\sum_{i=1}^{n}{(y_i - \\bar{y})^2}}}

    where:
        - :math:`\\rho` = Pearson's correlation coefficient
        - :math:`x_i` = the values of x in a sample (i.e. forecast values)
        - :math:`\\bar{x}` = the mean value of the forecast sample
        - :math:`y_i` = the values of y in a sample (i.e. observed values)
        - :math:`\\bar{y}` = the mean value of the observed sample value

    Args:
        fcst: Forecast or predicted variables
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Pearson's correlation coefficient.
            All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the Pearson's correlation coefficient. All other dimensions will
            be reduced. As a special case, 'all' will allow all dimensions to be
            preserved. In this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the absolute error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
    Returns:
        An xarray object with Pearson's correlation coefficient values

    Raises:
        ValueError: If a user tries to preserve all dimensions a ValueError will be raised.
        TypeError: If the input types are not xarray DataArrays or Datasets.
        ValueError: If the input Datasets do not have the same data variables.

    Note:
        This function isn't set up to take weights.

    Reference:
        https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    See Also:
        :py:func:`scores.continuous.correlation.spearmanr`

    Examples:
        >>> import xarray as xr
        >>> from scores.continuous.correlation.correlation_impl import pearsonr

        >>> times = [1, 2]
        >>> locations = ["A", "B", "C"]

        >>> fcst = xr.DataArray(
        ...     data=[[0.1, 10.0, 0.0], [0.4, 7.1, 6.5]],
        ...     coords={"time": times, "location": locations},
        ...     dims=["time", "location"],
        ... )

        >>> obs = xr.DataArray(
        ...     data=[[-3.4, 13.4, 0.1], [0.4, 10.2, 4.5]],
        ...     coords={"time": times, "location": locations},
        ...     dims=["time", "location"],
        ... )

        >>> pearsonr(fcst, obs, reduce_dims="location")
        <xarray.DataArray (time: 2)> Size: 16B
        array([0.97856011, 0.85946704])
        Coordinates:
          * time     (time) int64 16B 1 2
    """
    if not all_same_xarraylike([fcst, obs]):
        raise TypeError("Both fcst and obs must be either xarray DataArrays or xarray Datasets.")

    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    if len(reduce_dims) == 0:
        raise ValueError("You cannot preserve all dimensions with pearsonr.")

    if isinstance(fcst, xr.DataArray) and isinstance(obs, xr.DataArray):
        return xr.corr(fcst, obs, reduce_dims)

    # Ensure both datasets have the same variables
    if set(fcst.data_vars) != set(obs.data_vars):
        raise ValueError("Both datasets must contain the same variables.")

    results = {_var: xr.corr(fcst[_var], obs[_var], reduce_dims) for _var in fcst.data_vars}

    return xr.Dataset(results)


def spearmanr(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
) -> XarrayLike:
    """
    Calculates the Spearman's rank correlation coefficient between two xarray objects.

    .. math::
        r_s = \\rho\\big(R(x), R(y)\\big)

    where:
        - :math:`\\rho` is the Pearson correlation coefficient.
        - :math:`R` is the ranking operator.

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Spearman's rank correlation coefficient.
            All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the Spearman's rank correlation coefficient. All other dimensions will
            be reduced. As a special case, 'all' will allow all dimensions to be
            preserved.

    Returns:
        An xarray object with Spearman's rank correlation coefficient values.

    Raises:
        ValueError: If a user tries to preserve all dimensions ValueError will be raised.
        TypeError: If the input types are not xarray DataArrays or Datasets.
        ValueError: If the input Datasets do not have the same data variables.

    Note:
        This function isn't set up to take weights.

    Reference:
        Spearman, C. (1904). The Proof and Measurement of Association between Two Things. The American Journal of
        Psychology, 15(1), 72–101. https://doi.org/10.2307/1412159

    See also:
        :py:func:`scores.continuous.correlation.pearsonr`

    Examples:
        >>> import xarray as xr
        >>> from scores.continuous.correlation.correlation_impl import spearmanr

        >>> times = [1, 2]
        >>> locations = ["A", "B", "C"]

        >>> fcst = xr.DataArray(
        ...     data=[[0.1, 10.0, 0.0], [0.4, 7.1, 6.5]],
        ...     coords={"time": times, "location": locations},
        ...     dims=["time", "location"],
        ... )

        >>> obs = xr.DataArray(
        ...     data=[[-3.4, 13.4, 0.1], [0.4, 10.2, 4.5]],
        ...     coords={"time": times, "location": locations},
        ...     dims=["time", "location"],
        ... )

        >>> spearmanr(fcst, obs, reduce_dims="location")
        <xarray.DataArray (time: 2)> Size: 16B
        array([0.5, 1. ])
        Coordinates:
          * time     (time) int64 16B 1 2
    """
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    if len(reduce_dims) == 0:
        raise ValueError("You cannot preserve all dimensions with spearmanr.")
    tmp_dim = "".join(list(fcst.dims) + list(obs.dims))
    obs_stacked = obs.stack({tmp_dim: reduce_dims})
    fcst_stacked = fcst.stack({tmp_dim: reduce_dims})
    # Rank
    fcst_ranks = fcst_stacked.rank(dim=tmp_dim)
    obs_ranks = obs_stacked.rank(dim=tmp_dim)

    return pearsonr(fcst_ranks, obs_ranks, reduce_dims=tmp_dim)
