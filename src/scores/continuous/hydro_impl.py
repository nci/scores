import functools
import warnings
from collections.abc import Hashable
from typing import Iterable, Literal, Optional

import numpy as np
import xarray as xr

import scores.continuous
from scores.processing import aggregate, broadcast_and_match_nan
from scores.typing import (
    FlexibleDimensionTypes,
    XarrayLike,
)
from scores.utils import gather_dimensions, validate_inputs_outputs


def merge_sizes(*ds) -> dict[Hashable, int]:
    """
    Merges the maps that contain the size (value) for each dimension (key)
    in the given ``xr.Dataset`` object(s).

    Args:
        *ds: Variadic argument of each of type ``xr.Dataset``
    """
    assert len(ds) > 0

    def _merge_single(
        acc_sizes: dict[Hashable, int],
        curr_ds: xr.Dataset | None,
    ) -> dict[Hashable, int]:
        """
        merges ``sizes`` attribute from each dataset. ``sizes`` is a
        mapping: dimension name -> length. In theory this should also be
        compatible with data arrays.
        """
        if curr_ds is None:
            return acc_sizes
        merged_sizes: dict[Hashable, int] = acc_sizes | dict(curr_ds.sizes)
        return merged_sizes

    ret_sizes: dict[Hashable, int] = functools.reduce(_merge_single, ds, {})

    return ret_sizes


@validate_inputs_outputs(same_input_types=True, same_input_and_output_type=True)
def nse(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: FlexibleDimensionTypes | None = None,
    preserve_dims: FlexibleDimensionTypes | None = None,
    weights: XarrayLike | None = None,
    is_angular: bool = False,
) -> XarrayLike:
    """
    The Nash-Sutcliffe model Efficiency coefficient (NSE) is primarily used in
    hydrology to assess the skill of model predictions (of e.g. "discharge").

    While NSE is often calculated over observations and model predictions in
    the time dimension, it is actually a fairly generic statistical measure
    that determines the relative magnitude of the residual variance ("noise")
    compared to the measured data variance ("information") (Nash and Sutcliffe,
    1970). Incidentally, it is (inversely) related to the signal-to-noise ratio
    (SNR).

    The general formulation of NSE is as follows:

    .. math::

        \\text{NSE} = 1 - \\frac{\\sum_i{(O_i - S_i)^2}}{\\sum_i{(O_i - \\bar{O})^2}}

    where:
        - :math:`i` is a generic "indexer" representing the set of datapoints
          along the dimensions being reduced e.g. time (:math:`t`) or
          xy-coordinates (:math:`(x, y)`). The latter represents reduction over
          two dimensions as an example.
        - :math:`O_i` is the observation at index :math:`i`.
        - :math:`S_i` is the "forecast" or model simulation at index :math:`i`.
        - :math:`\\bar{O}` is the mean observation of the set of indexed
          samples as specified by ``reduce_dims`` and ``preserve_dims``.

    Args:

        fcst: "Forecast" or predicted variables.

        obs: Observed variables.

        reduce_dims: dimensions to reduce when calculating the NSE. (i.e. NSE
            will be calculated using datapoints along these dimensions as
            samples, the other dimensions will be preserved).

        preserve_dims: dimensions to preserve. Mutually exclusive to
            ``reduce_dims``. All dimensions not specified here will be reduced
            as described in ``reduce_dims``.  Note: ``preserve_dims="all"`` is
            not supported for NSE. See notes below.

        weights: Optional weighting to apply to the NSE computation. Typically
            weights are applied over the time dimension but can vary by
            location as well. Weights must be non-negative and specified for
            each data point *(i.e. the user must not assume broadcasting will
            handle appropriate assignment of weights for this score)*.

        is_angular: specifies whether ``fcst`` and ``obs`` are angular data
            (e.g. wind direction).  If True, a different function is used to
            calculate the difference between ``fcst`` and ``obs``, which
            accounts for circularity. Angular ``fcst`` and ``obs`` data should
            be in degrees rather than radians.

    Returns:

        NSE score for each preserved dimension

        ``xr.Dataset``:  if ``fcst``, ``obs`` and optionally ``weights`` are all datasets.

        ``xr.DataArray``: ditto above - where inputs are all dataarrays

        See comments below for more information on mixed xarray data types
        (which this score does **not** handle)  and type isomorphism.

    Supplementary details:
        - Nash-Sutcliffe efficiencies range from -Inf to 1. Essentially, the
          closer to 1, the more accurate the model is.

          - NSE = 1, corresponds to a perfect match of the model to the obs.
          - NSE = 0, indicates that the model is as accurate as the mean obs.
          - -Inf < NSE < 0, indicates that the mean obs is better predictor than the model.

        - The optional ``weights`` argument can additionally be used to perform
          a weighted NSE (wNSE). Although, this is a generic set of weights,
          and it is the *user's responsiblility* to define them appropriately.
          Typically this is the observation itself
          (Hundecha, Y., & Bárdossy, A., 2004).
        - ``weights`` must be non-negative. Therefore, the observations must
          ideally also be non-negative (or formulated appropriately) if used as
          weights.
        - While ``is_angular`` is typically not used for this score, NSE is
          generic enough that it _could_ be used in wider context, and hence is
          kept as an option. It is defaulted to ``False`` as that's the typical
          use-case.

    .. important::

        This score does not allow mixed xarray data structures as inputs.
        Either provide all ``xr.DataArray`` or all ``xr.Dataset`` exclusively,
        for the ``fcst``, ``obs`` and (optionally) ``weights`` arguments.

        This is an intentionally imposed constraint to make sure the inner
        computations are simple to check and deterministic. See tips below for
        more information.

    .. warning::

        Operations between dataarrays are not guaranteed to preserve names. If
        the user is working with dataarrays, it is assumed that preserving
        names is not a major requirement. If a user needs the name preserved,
        they should explicitly convert all data array inputs to datasets using
        ``xr.DataArray.to_dataset(...)`` , and *verify* that the naming is
        retained appropriately before calling the score.

        For operations where ONLY ``xr.DataArray`` inputs are used, the
        returned score will have its name forced to the name of this score i.e.
        "NSE", for simplicity.

        See tips below for more information.

    .. note::

        For Hydrology in particular :math:`i = t`,  the reduced dimension is
        usually the time dimension. However, in order to keep things generic,
        this function does not explicitly mandate a time dimension be provided.
        Instead it requires it has a hard requirement that *at least one*
        dimension is being reduced from either a specification of
        ``reduce_dims`` or ``preserve_dims`` (mutually exclusive).

        The reason is that the observation variance cannot be non-zero if
        nothing is being reduced.

        As a side-effect of the above requirement, ``preserve_dims="all"`` is
        not allowed and will naturally throw an error.

    .. note::

        Divide by zero *is allowed* - to accomodate scenarios where all obs
        entries in the group being reduced is constant (0 obs variance).

        While these may cause divide by zero warnings, they should not halt
        execution of computations for other valid coordinates - so a warning is
        issued instead to prompt the user to double check the data.

        It may also be that divide by zero is unavoidable - in which case we
        still want to return the correctly calculated values. To this end, this
        is how ``numpy`` resolves divide by zero:

        .. code-block::

            np.divide(n, 0) = np.nan    # if n == 0
                            = np.inf    # if n > 0
                            = -np.inf   # if n > 0


    .. tip::

        When dealing with dask arrays dask, no computation will happen until
        ``.compute(...)`` is called on the returned score.

    .. tip::

        Work with datasets where possible with NSE, or for any score that
        supports datasets for that matter. Datasets maintain structural
        integrity better than their dataarray counterparts and also are
        compatible with higher order types like ``xr.DataTree``.

        Operations between datasets are more predictable than operations with
        mixed types. Data arrays on the other hand may ignore names and
        broadcast liberally even when names do not match, and this may not be
        consistent depending on the operation. This may or may not be the
        intented behaviour the user expects. Operations between **only**
        dataarrays are fine as long as preserving names is not mandatory.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.continuous import nse

        >>> obs_raw = np.array(
        ...     [
        ...         [[1, 2, 3], [4, 5, 6]],
        ...         [[3, 2, 1], [6, 5, 4]],
        ...         [[3, 2, 5], [2, 2, 6]],
        ...         [[5, 2, 3], [4, -1, 4]],
        ...     ]
        ... )  # dimension lengths: x=4, y=2, t=3
        >>> obs = xr.DataArray(obs_raw, dims=["x", "y", "t"])

        >>> # add some synthetic bias and variance
        >>> fcst = obs * 1.2 + 0.1

        >>> # reduce over t - time - should produce a xy-grid (4 by 2)
        >>> nse(obs, fcst, reduce_dims=["t"])
        <xarray.DataArray 'NSE' (x: 4, y: 2)> Size: 64B
        array([[ 0.71180556, -0.28819444],
               [ 0.71180556, -0.28819444],
               [ 0.70982143,  0.85742188],
               [ 0.70982143,  0.93208333]])
        Dimensions without coordinates: x, y

        >>> # reduce over (x, y) - space - should be a t-vector (3 by 1)
        >>> nse(obs, fcst, reduce_dims=["x", "y"])
        <xarray.DataArray 'NSE' (t: 3)> Size: 24B
        array([0.77469136, 0.90123457, 0.74722222])
        Dimensions without coordinates: t

    References:

        1. Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting
           through conceptual models part I — A discussion of principles. In
           Journal of Hydrology (Vol. 10, Issue 3, pp. 282– 290). Elsevier BV.
           https://doi.org/10.1016/0022-1694%2870%2990255-6

        2. Hundecha, Y., & Bárdossy, A. (2004). Modeling of the effect of land
           use changes on the runoff generation of a river basin through parameter
           regionalization of a watershed model. Journal of Hydrology, 292(1-4),
           281-295. https://doi.org/10.1016/j.jhydrol.2004.01.002
    """
    fcst, obs = broadcast_and_match_nan(fcst, obs)

    gathered_dims: FlexibleDimensionTypes = gather_dimensions(
        fcst_dims=fcst.dims,
        obs_dims=obs.dims,
        weights_dims=weights.dims if weights is not None else None,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

    if len(list(gathered_dims)) == 0:
        raise ValueError("No dimensions to reduce. NSE requires at least one dimension to reduce")
    dim_sizes = merge_sizes(obs, fcst, weights)
    dim_has_more_than_one_obs = any(dim_sizes[k] > 1 for k in gathered_dims)

    if not dim_has_more_than_one_obs:
        raise ValueError("Do not have more than one observation.")

    fcst_error = scores.continuous.mse(
        fcst,
        obs,
        reduce_dims=gathered_dims,
        weights=weights,
        is_angular=is_angular,
    )
    obs_mean = obs.mean(dim=gathered_dims)
    obs_variance = scores.continuous.mse(
        obs_mean,
        obs,
        reduce_dims=gathered_dims,
        weights=weights,
        is_angular=is_angular,
    )

    is_zero = obs_variance == 0
    if isinstance(is_zero, xr.Dataset):
        has_zero = any(bool(v.any()) for v in is_zero.data_vars.values())
    else:
        has_zero = bool(is_zero.any())
    if has_zero:
        warnings.warn(
            "divide by zero encountered in NSE calculation",
            RuntimeWarning,
            stacklevel=2,
        )

    nse = 1.0 - (fcst_error / obs_variance)

    if isinstance(nse, xr.DataArray):
        nse.name = "NSE"

    return nse


@validate_inputs_outputs()
def kge(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    scaling_factors: Optional[Iterable[float]] = None,
    include_components: Optional[bool] = False,
    method: Literal["2009", "2012"] = "2009",
) -> XarrayLike:
    # pylint: disable=too-many-locals
    """
    Calculate the Kling-Gupta Efficiency (KGE) between observed and simulated (or forecast) values.

    KGE is a performance metric that decomposes the error into three components:
    correlation, variability, and bias.

    .. math::
        \\alpha = \\frac{\\sigma_x}{\\sigma_y}

    .. math::
        \\beta = \\frac{\\mu_x}{\\mu_y}

    .. math::
        \\gamma = \\frac{\\alpha}{\\beta}

    .. math::
        {vr} = \\alpha \\text{ if original 2009 KGE else } \\gamma \\text{ (modified 2012 KGE)}

    The KGE is computed as

    .. math::
        \\text{KGE} = 1 - \\sqrt{\\left[s_\\rho \\cdot (\\rho - 1)\\right]^2 +
        \\left[s_{vr} \\cdot ({vr} - 1)\\right]^2 + \\left[s_\\beta \\cdot (\\beta - 1)\\right]^2}

    where:
        - :math:`\\rho`  = Pearson's correlation coefficient between observed and forecast values as
          defined in :py:func:`scores.continuous.correlation.pearsonr`
        - :math:`\\alpha` is the ratio of the standard deviations (variability ratio)
        - :math:`\\gamma` is the ratio of the coefficients of variation (relative variability ratio/coefficient of variation ratio)
        - :math:`\\beta` is the ratio of the means (bias)
        - :math:`x` and :math:`y` are forecast and observed values, respectively
        - :math:`\\mu_x` and :math:`\\mu_y` are the means of forecast and observed values, respectively
        - :math:`\\sigma_x` and :math:`\\sigma_y` are the standard deviations of forecast and observed values, respectively
        - :math:`s_\\rho`, :math:`s_{vr}`, and :math:`s_\\beta` are the scaling factors for the correlation coefficient :math:`\\rho`,
          the variability term :math:`{vr}`, and the bias term :math:`\\beta`

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the KGE. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the KGE. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be all NaN with the same shape/dimensionality
            as the forecast because the standard deviation is zero for a single point.
        scaling_factors: A 3-element vector or list describing the weights for each term in the KGE.
            Defined by: scaling_factors = [:math:`s_\\rho`, :math:`s_{vr}`, :math:`s_\\beta`] to apply to the correlation term :math:`\\rho`,
            the variability term :math:`{vr}` and the bias term :math:`\\beta` respectively. Defaults to (1.0, 1.0, 1.0).
        include_components (bool | False): If True, the function also returns the individual terms contributing to the KGE score.
        method: Whether to compute the original KGE as defined in Gupta et al. (2009) or the modified KGE as defined in Kling et al. (2012).
            Default is "2009".

    Returns:
        If ``include_components`` is False, the function returns the KGE score as an ``xarray.DataArray``.

        If ``include_components`` is True, the function returns ``xarray.Dataset`` with the following variables:

        - `kge`: The KGE score.
        - `rho`: The Pearson correlation coefficient.
        - (if original 2009 KGE) `alpha`: The variability ratio.
        - (if modified 2012 KGE) `gamma`: The coefficient of variation ratio.
        - `beta`: The bias term.

    Notes:
        - Statistics are calculated only from values for which both observations and
          simulations are not null values.
        - This function isn't set up to take weights.
        - Currently this function is working only on ``xarray.DataArray``.
        - When preserve_dims is set to 'all', the function returns NaN,
          similar to the Pearson correlation coefficient calculation for a single data point
          because the standard deviation is zero for a single point.

    References:
        -   Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and
            NSE performance criteria: Implications for improving hydrological modeling. Journal of Hydrology, 377(1-2), 80-91.
            https://doi.org/10.1016/j.jhydrol.2009.08.003.
        -   Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under an ensemble of climate
            change scenarios. Journal of Hydrology. 424: 264–277. https://doi.org/10.1016/j.jhydrol.2012.01.011.
        -   Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not?
            Comparing Nash-Sutcliffe and Kling-Gupta efficiency scores. Hydrology and Earth System Sciences, 23(10), 4323-4331.
            https://doi.org/10.5194/hess-23-4323-2019.


    Examples:
        >>> import xarray as xr
        >>> from scores.continuous import kge
        >>> from scores.functions import create_latitude_weights

        >>> times = ["2024-01-01", "2024-01-02"]
        >>> lats = [-35, -30, -25]
        >>> lons = [140, 150]

        >>> obs = xr.DataArray(
        ...     [
        ...         [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
        ...         [[1.5, 1.6], [2.5, 2.6], [3.5, 3.6]],
        ...     ],
        ...     coords={"time": times, "lat": lats, "lon": lons},
        ...     dims=["time", "lat", "lon"],
        ... )

        >>> fcst = xr.DataArray(
        ...     [
        ...         [[1.3, 1.0], [1.7, 2.7], [3.6, 3.0]],
        ...         [[1.0, 1.1], [2.2, 2.7], [3.3, 3.9]],
        ...     ],
        ...     coords={"time": times, "lat": lats, "lon": lons},
        ...     dims=["time", "lat", "lon"],
        ... )

        >>> weights = xr.DataArray(
        ...     create_latitude_weights(lats), coords={"lat": lats}, dims=["lat"]
        ... )

        >>> kge(fcst, obs)
        <xarray.DataArray ()> Size: 8B
        array(0.80729437)

        >>> kge(fcst, obs, preserve_dims=["time"])
        <xarray.DataArray (time: 2)> Size: 16B
        array([0.8118117 , 0.68668803])
        Coordinates:
          * time     (time) <U10 80B '2024-01-01' '2024-01-02'

        >>> kge(fcst, obs, scaling_factors=[0.5, 1.0, 1.5])
        <xarray.DataArray ()> Size: 8B
        array(0.81581987)

        >>> kge(fcst, obs, include_components=True)
        <xarray.Dataset> Size: 32B
        Dimensions:  ()
        Data variables:
            kge      float64 8B 0.8073
            rho      float64 8B 0.9344
            alpha    float64 8B 1.181
            beta     float64 8B 0.9964

        >>> kge(fcst, obs, include_components=True, method="2012")
        <xarray.Dataset> Size: 32B
        Dimensions:  ()
        Data variables:
            kge      float64 8B 0.8033
            rho      float64 8B 0.9344
            gamma    float64 8B 1.185
            beta     float64 8B 0.9964

    """  # noqa: E501

    # Type checks as xrray.corr can only handle xr.DataArray
    if not isinstance(fcst, xr.DataArray):
        raise TypeError("kge: fcst must be an xarray.DataArray")
    if not isinstance(obs, xr.DataArray):
        raise TypeError("kge: obs must be an xarray.DataArray")
    if method not in ["2009", "2012"]:
        raise ValueError("kge: method must be either '2009' or '2012'")
    if scaling_factors is None:
        scaling_factors = (1.0, 1.0, 1.0)
    try:
        s_rho, s_vr, s_beta = scaling_factors
    except ValueError as e:
        raise ValueError("kge: scaling_factors must be an iterable of exactly 3 elements") from e

    reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)
    # Need to broadcast and match NaNs so that the fcst and obs are for the
    # same points
    fcst, obs = broadcast_and_match_nan(fcst, obs)
    # compute linear correlation coefficient r between fcst and obs
    rho = xr.corr(fcst, obs, reduce_dims)

    # compute alpha (sigma_sim / sigma_obs)
    sigma_fcst = fcst.std(reduce_dims)
    sigma_obs = obs.std(reduce_dims)
    alpha = sigma_fcst / sigma_obs

    # compute beta (mu_sim / mu_obs)
    mu_fcst = fcst.mean(reduce_dims)
    mu_obs = obs.mean(reduce_dims)
    beta = mu_fcst / mu_obs

    vr = alpha if method == "2009" else alpha / beta

    ed_s = np.sqrt((s_rho * (rho - 1)) ** 2 + (s_vr * (vr - 1)) ** 2 + (s_beta * (beta - 1)) ** 2)

    kge_s = 1 - ed_s

    if include_components:
        # Create dataset of all components
        vr_name = "alpha" if method == "2009" else "gamma"
        component_names = ["kge", "rho", vr_name, "beta"]
        components = [kge_s, rho, vr, beta]
        kge_dict = dict(zip(component_names, components))
        kge_s = xr.Dataset(kge_dict)
    return kge_s


@validate_inputs_outputs()
def pbias(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
) -> XarrayLike:
    """
    Calculates the percent bias, which is the ratio of the additive bias to the mean observed value, multiplied by 100.

    Percent bias is used for evaluating and comparing forecast accuracy across stations or datasets with varying
    magnitudes. By expressing the error as a percentage of the observed value, it allows for standardised comparisons,
    enabling assessment of forecast performance regardless of the absolute scale of values. Like
    :py:func:`scores.continuous.multiplicative_bias`, ``pbias`` will return a ``np.inf`` where the mean of ``obs``
    across the dims to be reduced is 0. It is defined as

    .. math::
        \\text{Percent bias} = 100 \\cdot \\frac{\\sum_{i=1}^{N}(x_i - y_i)}{\\sum_{i=1}^{N} y_i}

    where:
        - :math:`x_i` = the values of x in a sample (i.e. forecast values)
        - :math:`y_i` = the values of y in a sample (i.e. observed values)

    See "pbias" section at https://search.r-project.org/CRAN/refmans/hydroGOF/html/pbias.html for more information

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the percent bias. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the percent bias. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: An array of weights to apply to the score (e.g., weighting a grid by latitude).
            If None, no weights are applied. If provided, the weights must be broadcastable
            to the data dimensions and must not contain negative or NaN values. If
            appropriate, NaN values in weights  can be replaced by ``weights.fillna(0)``.
            The weighting approach follows :py:class:`xarray.computation.weighted.DataArrayWeighted`.
            See the scores weighting tutorial for more information on how to use weights.

    Returns:
        An xarray object with the percent bias of a forecast.

    References:
        -   Sorooshian, S., Duan, Q., & Gupta, V. K. (1993). Calibration of rainfall-runoff models:
            Application of global optimization to the Sacramento Soil Moisture Accounting Model.
            Water Resources Research, 29(4), 1185-1194. https://doi.org/10.1029/92WR02617
        -   Alfieri, L., Pappenberger, F., Wetterhall, F., Haiden, T., Richardson, D., & Salamon, P. (2014).
            Evaluation of ensemble streamflow predictions in Europe. Journal of Hydrology, 517, 913-922.
            https://doi.org/10.1016/j.jhydrol.2014.06.035
        -   Dawson, C. W., Abrahart, R. J., & See, L. M. (2007). HydroTest:
            A web-based toolbox of evaluation metrics for the standardised assessment of hydrological forecasts.
            Environmental Modelling and Software, 22(7), 1034-1052.
            https://doi.org/10.1016/j.envsoft.2006.06.008
        -   Moriasi, D. N., Arnold, J. G., Van Liew, M. W., Bingner, R. L., Harmel, R. D., & Veith, T. L. (2007).
            Model evaluation guidelines for systematic quantification of accuracy in watershed simulations.
            Transactions of the ASABE, 50(3), 885-900. https://doi.org/10.13031/2013.23153



    Examples:
        >>> import xarray as xr
        >>> from scores.continuous import pbias
        >>> from scores.functions import create_latitude_weights

        >>> times = ["2024-01-01", "2024-01-02"]
        >>> lats = [-35, -30, -25]
        >>> lons = [140, 150]

        >>> obs = xr.DataArray(
        ...     [
        ...         [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
        ...         [[1.5, 1.6], [2.5, 2.6], [3.5, 3.6]],
        ...     ],
        ...     coords={"time": times, "lat": lats, "lon": lons},
        ...     dims=["time", "lat", "lon"],
        ... )

        >>> fcst = xr.DataArray(
        ...     [
        ...         [[1.3, 1.0], [1.7, 2.7], [3.6, 3.0]],
        ...         [[1.0, 1.1], [2.2, 2.7], [3.3, 3.9]],
        ...     ],
        ...     coords={"time": times, "lat": lats, "lon": lons},
        ...     dims=["time", "lat", "lon"],
        ... )

        >>> weights = xr.DataArray(
        ...     create_latitude_weights(lats), coords={"lat": lats}, dims=["lat"]
        ... )

        >>> pbias(fcst, obs)
        <xarray.DataArray ()> Size: 8B
        array(-0.36231884)

        >>> pbias(fcst, obs, preserve_dims=["time"])
        <xarray.DataArray (time: 2)> Size: 16B
        array([ 8.1300813 , -7.18954248])
        Coordinates:
          * time     (time) <U10 80B '2024-01-01' '2024-01-02'

        >>> pbias(fcst, obs, weights=weights)
        <xarray.DataArray ()> Size: 8B
        array(-0.10307618)

    """
    reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)
    # Need to broadcast and match NaNs so that the mean error and obs mean are for the
    # same points
    fcst, obs = broadcast_and_match_nan(fcst, obs)
    error = fcst - obs

    numerator = 100 * aggregate(error, reduce_dims=reduce_dims, weights=weights)
    denominator = aggregate(obs, reduce_dims=reduce_dims, weights=weights)
    _pbias = numerator / denominator
    return _pbias
