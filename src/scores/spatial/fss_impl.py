"""
This module contains methods related to the Fractions Skill Score (FSS).

For an explanation of the FSS, and implementation considerations, see references below.

Currently uses the default score defined Robert and Leans (2008) :sup:`[1,2]`. Fine grained controls are considered in `#353 <GITHUB353_>`_.

The default computation is performed using`numpy` :sup:`[3]` and summed-area tables, with future optimization options considered in:
- `#269 <GITHUB269_>`_.
- `#270 <GITHUB270_>`_.

References:
1. Roberts, N. M., and H. W. Lean, 2008: Scale-Selective Verification of Rainfall Accumulations from
   High-Resolution Forecasts of Convective Events. Monthly Weather Review, 136, 78–97,
   https://doi.org/10.1175/2007mwr2123.1.
2. Mittermaier, M. P., 2021: A “Meta” Analysis of the Fractions Skill Score: The Limiting Case and
   Implications for Aggregation. Monthly Weather Review, 149, 3491–3504,
   https://doi.org/10.1175/mwr-d-18-0106.1.
3. FAGGIAN, N., B. ROUX, P. STEINLE, and B. EBERT, 2015: Fast calculation of the fractions skill
   score. MAUSAM, 66, 457–466, https://doi.org/10.54302/mausam.v66i3.555.

.. _GITHUB269: https://github.com/nci/scores/issues/269
.. _GITHUB270: https://github.com/nci/scores/issues/270
.. _GITHUB353: https://github.com/nci/scores/issues/353
"""

# sphinx docstrings: issue with inconsistent type representations
# see: https://github.com/sphinx-doc/sphinx/issues/9813
from __future__ import annotations  # isort: off

import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from numpy import typing as npt

from scores.fast.fss.fss_backends import get_compute_backend
from scores.fast.fss.typing import FssComputeMethod, FssDecomposed
from scores.processing import broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes
from scores.utils import (
    DimensionError,
    DimensionWarning,
    FieldTypeError,
    NumpyThresholdOperator,
    gather_dimensions,
    left_identity_operator,
)

# Set the seed for reproducibility
np.random.seed(42)


def _fss_2d_without_ref(  # pylint: disable=too-many-locals,too-many-arguments
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    event_threshold: Optional[float],
    window_size: Tuple[int, int],
    spatial_dims: Tuple[str, str],
    padding: Optional[str] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    threshold_operator: Optional[Callable] = None,
    compute_method: FssComputeMethod = FssComputeMethod.NUMPY,
    dask: str = "forbidden",  # see: `xarray.apply_ufunc` for options
) -> xr.Dataset:
    """
    Computes the Fractions Skill Score (FSS) for each 2D spatial field in the DataArray and then
    aggregates them over the output of gather dimensions. It does not include FSS for a reference
    forecast.

    The aggregation method is the intended extension of the score defined by Robert and Leans (2008)
    for multiple forecasts :sup:`[1,2]`

    .. note::
        This method takes in a ``threshold_operator`` to compare the input
        fields against the ``event_threshold`` which defaults to ``numpy.greater``,
        and is compatible with lightweight numpy operators.

    Optionally aggregates the output along other dimensions if `reduce_dims` or
    (mutually exclusive) ``preserve_dims`` are specified.

    For implementation for a single 2-D field,
    see: :py:func:`fss_2d_single_field`. This offers the user the ability to use
    their own aggregation methods, rather than ``xarray.ufunc``

    .. warning::
        ``dask`` option is not fully tested

    Args:
        fcst: An array of forecasts
        obs: An array of observations (same spatial shape as ``fcst``)
        event_threshold: A scalar to compare ``fcst`` and ``obs`` fields to generate a
            binary "event" field. (defaults to ``np.greater``).
        window_size: A pair of positive integers ``(height, width)`` of the sliding
            window_size; the window size must be greater than 0 and fit within
            the shape of ``obs`` and ``fcst``.
        spatial_dims: A pair of dimension names ``(x, y)`` where ``x`` and ``y`` are
            the spatial dimensions to slide the window along to compute the FSS.
            e.g. ``("lat", "lon")``.
        padding: To handle the edge points, user can use either zero or reflective padding.
            If set to ``"zero"``, applies a 0-valued window border around the
            data field before computing the FSS. If set to ``"reflective"``, the values at the
            edges are mirrored outward. If set to None (default), it uses the edges
            (thickness = window size) of the input fields as the border.
            One can think of it as:
            - padding = None => inner border
            - padding = "zero" => outer border with 0 values
            - padding = "reflective" => outer border with mirrored values
        reduce_dims: Optional set of additional dimensions to aggregate over
            (mutually exclusive to ``preserve_dims``).
        preserve_dims: Optional set of dimensions to keep (all other dimensions
            are reduced); By default all dimensions except ``spatial_dims``
            are preserved (mutually exclusive to ``reduce_dims``).
        threshold_operator: The threshold operator used to generate the binary
            event field. E.g. ``np.greater``. Note: this may depend on the backend
            ``compute method`` and not all operators may be supported. Generally,
            ``np.greater``, ``np.less`` and their counterparts. Defaults to "np.greater".
        compute_method: currently only supports :py:obj:`FssComputeMethod.NUMPY`
            see: :py:class:`scores.fast.fss.typing.FssComputeMethod`
        dask: See ``xarray.apply_ufunc`` for options

    Returns:
        An ``xarray.Dataset`` containing the FSS and its decompositions (i.e., Fractions
        Brier Score (FBS) and reference FBS (FBS_ref) computed over the ``spatial_dims``

        The resultant dataset will have the score grouped against the remaining
        dimensions, unless ``reduce_dims``/``preserve_dims`` are specified; in which
        case, they will be aggregated over the specified dimensions accordingly.

        For an exact usage please refer to ``Fractions_Skill_Score.ipynb`` in the tutorials.

    Raises:
        DimensionError: Various errors are thrown if the input dimensions
            do not conform. e.g. if the window size is larger than the input
            arrays, or if the the spatial dimensions in the args are missing in
            the input arrays.

        DimensionWarning: If ``spatial_dims`` are attempting to be preserved e.g. in ``preserve_dims``

    References:
        1. Roberts, N. M., and H. W. Lean, 2008: Scale-Selective Verification of Rainfall
           Accumulations from High-Resolution Forecasts of Convective Events. Monthly Weather
           Review, 136, 78–97, https://doi.org/10.1175/2007mwr2123.1.
        2. Mittermaier, M. P., 2021: A “Meta” Analysis of the Fractions Skill Score: The Limiting
           Case and Implications for Aggregation. Monthly Weather Review, 149, 3491–3504,
           https://doi.org/10.1175/mwr-d-18-0106.1.
    """
    np_thrsh_op = _make_numpy_threshold_operator(threshold_operator)

    def _spatial_dims_exist(_dims):
        s_spatial_dims = set(spatial_dims)
        s_dims = set(_dims)
        return (
            len(s_spatial_dims - s_dims) == 0
            and len(spatial_dims) == 2  # all spatial dims present  # number of spatial dims = 2
        )

    if not _spatial_dims_exist(fcst.dims):
        raise DimensionError(f"missing spatial dims {spatial_dims} in fcst")
    if not _spatial_dims_exist(obs.dims):
        raise DimensionError(f"missing spatial dims {spatial_dims} in obs")
    for s_dim in spatial_dims:
        if obs[s_dim].shape != fcst[s_dim].shape:
            raise DimensionError(
                f"The spatial extent of fcst and obs do not match for the following spatial dimensions: {spatial_dims}."
            )

    # wrapper defined for convenience, since it's too long for a lambda.
    def _fss_wrapper(da_fcst: xr.DataArray, da_obs: xr.DataArray) -> FssDecomposed:
        fss_backend = get_compute_backend(compute_method)
        fb_obj = fss_backend(
            da_fcst,
            da_obs,
            event_threshold=event_threshold,
            window_size=window_size,
            padding=padding,
            threshold_operator=np_thrsh_op,
        )
        return fb_obj.compute_fss_decomposed()

    # apply ufunc to get the decomposed fractions skill score aggregated over
    # the 2D spatial dims.
    da_fss = xr.apply_ufunc(
        _fss_wrapper,
        fcst,
        obs,
        input_core_dims=[list(spatial_dims), list(spatial_dims)],
        vectorize=True,
        dask=dask,  # type: ignore # pragma: no cover
    )

    # gather additional dimensions to aggregate over.
    dims = gather_dimensions(
        fcst.dims,
        obs.dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

    if not (spatial_dims[0] in dims and spatial_dims[1] in dims):
        # at least one spatial dim is trying to be preserved.
        warnings.warn(
            f"At least one of the provided spatial dims = {spatial_dims} is attempting "
            "to be preserved. Please make sure `preserve_dims` and `reduce_dims` "
            "are configured to not preserve `spatial_dims`, in order to suppress this "
            "warning.",
            DimensionWarning,
        )

    # trim any spatial dimensions as they shouldn't appear.
    dims = set(dims) - set(spatial_dims)

    # apply ufunc again but this time to compute the fss, reducing
    # any non-spatial dimensions.
    aggregated = xr.apply_ufunc(
        _aggregate_fss_decomposed,
        da_fss,
        input_core_dims=[list(dims)],
        vectorize=True,
        dask=dask,
        output_dtypes=[object],  # type: ignore # pragma: no cover
    )
    fss = xr.apply_ufunc(lambda x: x["fss"], aggregated, vectorize=True)
    fbs = xr.apply_ufunc(lambda x: x["fbs"], aggregated, vectorize=True)
    fbs_ref = xr.apply_ufunc(lambda x: x["fbs_ref"], aggregated, vectorize=True)
    result = xr.Dataset(data_vars={"FSS": fss, "FBS": fbs, "FBS_ref": fbs_ref})

    return result  # type: ignore


def fss_2d(  # pylint: disable=too-many-locals,too-many-arguments
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    event_threshold: Optional[float] = None,
    window_size: Tuple[int, int],
    spatial_dims: Tuple[str, str],
    is_input_binary: bool = False,
    check_boolean: bool = False,
    benchmark: Optional[str] = None,
    padding: Optional[str] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    threshold_operator: Optional[Callable] = None,
    compute_method: FssComputeMethod = FssComputeMethod.NUMPY,
    dask: str = "forbidden",  # see: `xarray.apply_ufunc` for options
) -> xr.Dataset:
    """
    Computes the Fractions Skill Score (FSS). In addition to the FSS and its decompositions,
    it also returns the FSS for a reference forecast if `benchmark` is set to either `uniform`,
    `random` or `both`.

    Based on current literature, two key benchmarks are used to assess whether a forecast system
    is sufficiently skillful:
    (i) The most widely used benchmark is the FSS achieved at the grid scale from a forecast where
    each grid point has a fraction equal to the base rate (i.e., the observed frequency of occurrence)
    :sup:`[1,2]`. Users need to set ``benchmark`` to ``'uniform'`` or ``'both'`` to calculate the
    uniform FSS (UFSS).
    (ii) Recently proposed by Antonio and Aitchison (2025), this benchmark represents the FSS obtained
    from a random forecast. The random forecast is generated by sampling from a Bernoulli distribution
    at each grid point, with the probability set to the base rate :sup:`[3]`. Users need to set
    ``benchmark`` to ``'random'`` or ``'both'`` to calculate the reference score using a random forecast
    (FSS_rand).

    Please note that FSS values for both benchmarks will be included in the output dataset if
    ``benchmark='both'`` is specified by the user.

    By default, binary fields for forecast and observation are computed using numpy.greater with a
    specified ``event_threshold``. However, the function also accepts pre-discretized binary fields
    (i.e., arrays of True/False values). If you require more advanced binary discretization, it is
    recommended to perform this step separately and then pass the binary fields to the function with
    ``is_input_binary=True``.

    Optionally the user can set ``check_boolean`` to ``True`` to check that the
    field is boolean before any computations. Note: this asserts that the
    underlying fields are of type ``np.bool_`` but does not coerce them. If
    the user is confident that the input field is binary (but not necessarily
    boolean), they may set this flag to ``False``, to allow for more flexible binary
    configurations such as 0 and 1; this should give the same results.

    Args:
        fcst: An array of forecasts
        obs: An array of observations (same spatial shape as ``fcst``)
        event_threshold: A scalar to compare ``fcst`` and ``obs`` fields to generate a
            binary "event" field. (defaults to ``np.greater``).
        window_size: A pair of positive integers ``(height, width)`` of the sliding
            window_size; the window size must be greater than 0 and fit within
            the shape of ``obs`` and ``fcst``.
        spatial_dims: A pair of dimension names ``(x, y)`` where ``x`` and ``y`` are
            the spatial dimensions to slide the window along to compute the FSS.
            e.g. ``("lat", "lon")``.
        is_input_binary: True if a binary field of obs and fcst is supplied.
        check_boolean: True if the user want input fields to be checked are
            boolean before any computations.
        benchmark: Benchmark to calculate a reference forecast. Users can select
            either 'uniform', 'random', 'both' or None.
        padding: To handle the edge points, user can use either zero or reflective padding.
            If set to ``"zero"``, applies a 0-valued window border around the
            data field before computing the FSS. If set to ``"reflective"``, the values at the
            edges are mirrored outward. If set to None (default), it uses the edges
            (thickness = window size) of the input fields as the border.
            One can think of it as:
            - padding = None => inner border
            - padding = "zero" => outer border with 0 values
            - padding = "reflective" => outer border with mirrored values
        reduce_dims: Optional set of additional dimensions to aggregate over
            (mutually exclusive to ``preserve_dims``).
        preserve_dims: Optional set of dimensions to keep (all other dimensions
            are reduced); By default all dimensions except ``spatial_dims``
            are preserved (mutually exclusive to ``reduce_dims``).
        threshold_operator: The threshold operator used to generate the binary
            event field. E.g. ``np.greater``. Note: this may depend on the backend
            ``compute method`` and not all operators may be supported. Generally,
            ``np.greater``, ``np.less`` and their counterparts. Defaults to "np.greater".
        compute_method: currently only supports :py:obj:`FssComputeMethod.NUMPY`
            see: :py:class:`scores.fast.fss.typing.FssComputeMethod`
        dask: See ``xarray.apply_ufunc`` for options

    Returns:
        An ``xarray.DataArray`` containing the FSS and its decompositions (i.e., Fractions
        Brier Score (FBS) and reference FBS) computed over the ``spatial_dims``

        The resultant array will have the score grouped against the remaining
        dimensions, unless ``reduce_dims``/``preserve_dims`` are specified; in which
        case, they will be aggregated over the specified dimensions accordingly.

        For an exact usage please refer to ``Fractions_Skill_Score.ipynb`` in the tutorials.

    Raises:
        ValueError: If ``is_input_binary=True`` while no event threshold is defined.
        ValueError: If ``check_boolean=True`` while ``is_input_binary=False``.
        Field_TypeError: If ``check_boolean=True`` and the input fields are not boolean.

    References:
        1. Roberts, N. M., and H. W. Lean, 2008: Scale-Selective Verification of Rainfall
           Accumulations from High-Resolution Forecasts of Convective Events. Monthly Weather
           Review, 136, 78–97, https://doi.org/10.1175/2007mwr2123.1.
        2. Mittermaier, M. P., 2021: A “Meta” Analysis of the Fractions Skill Score: The Limiting
           Case and Implications for Aggregation. Monthly Weather Review, 149, 3491–3504,
           https://doi.org/10.1175/mwr-d-18-0106.1.
        3. Antonio, B. and Aitchison, L., 2025: How to derive skill from the Fractions Skill Score.
            Monthly Weather Review, 153(6), 1021-1033, https://doi.org/10.1175/MWR-D-24-0120.1.

    Example:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.spatial import fss_2d
        >>> fcst = xr.DataArray(np.random.rand(10, 20, 20), dims=['time', 'lat', 'lon'])
        >>> obs = xr.DataArray(np.random.rand(10, 20, 20), dims=['time', 'lat', 'lon'])
        >>> fss = fss_2d(
        ...     fcst,
        ...     obs,
        ...     event_threshold=0.1,
        ...     window_size=[9, 9],
        ...     spatial_dims=["lat", "lon"],
        ...     benchmark='both'
        ... )

    Uses ``_fss_2d_without_ref`` from the ``scores.spatial`` module to perform fss
    computation.
    """
    if not is_input_binary and event_threshold is None:
        raise ValueError("The event_threshold must be specified in the case where input fields are not binary.")
    if check_boolean and not is_input_binary:
        raise ValueError("The check_boolean can be set to True only if the is_input_binary is True.")

    if check_boolean and not (fcst.dtype == np.bool_ and obs.dtype == np.bool_):
        raise FieldTypeError("Input field is not boolean.")

    if is_input_binary:
        # Note: The event_threshold is set to a dummy value that will be discarded
        # by the `left_identity_operator`. This approach helps avoid passing "None"
        # to threshold operators, which may lead to undefined behaviour).
        event_threshold = -999
        threshold_operator = left_identity_operator
    result = _fss_2d_without_ref(  # pylint: disable=too-many-locals,too-many-arguments
        fcst,
        obs,
        event_threshold=event_threshold,
        window_size=window_size,
        spatial_dims=spatial_dims,
        padding=padding,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        threshold_operator=threshold_operator,
        compute_method=compute_method,
        dask=dask,  # see: `xarray.apply_ufunc` for options
    )
    if benchmark is None:
        return result
    else:
        np_thrsh_op = _make_numpy_threshold_operator(threshold_operator)
        dims = gather_dimensions(
            fcst.dims,
            obs.dims,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )
        # Broadcasting observations to match the dimensions of fcst, enabling the
        # calculation of FSS for a reference forecast and its inclusion in the output dataset
        _, br_obs = broadcast_and_match_nan(fcst, obs)
        br_obs_pop = np_thrsh_op.get()(br_obs, event_threshold)
        if benchmark in ["uniform", "both"]:
            ufss = (br_obs_pop.mean(dim=dims) / 2) + 0.5
            result = result.assign(UFSS=ufss)
        if benchmark in ["random", "both"]:
            obs_pr = np_thrsh_op.get()(br_obs, event_threshold).mean()
            rand_fcst = obs.copy()
            rand_fcst.data = np.random.binomial(n=1, p=obs_pr, size=obs.shape)
            fss_rand = _fss_2d_without_ref(
                rand_fcst,
                br_obs_pop,
                event_threshold=-999,
                window_size=window_size,
                spatial_dims=spatial_dims,
                padding=padding,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                threshold_operator=left_identity_operator,
                compute_method=compute_method,
                dask=dask,  # see: `xarray.apply_ufunc` for options
            )
            result = result.assign(FSS_rand=fss_rand["FSS"])
    return result


def fss_2d_single_field(
    fcst: npt.NDArray[float],  # type: ignore
    obs: npt.NDArray[float],
    *,
    event_threshold: float,
    window_size: Tuple[int, int],
    padding: Optional[str] = None,
    threshold_operator: Optional[Callable] = None,
    compute_method: FssComputeMethod = FssComputeMethod.NUMPY,
) -> Dict[str, np.float64]:
    """
    Calculates the Fractions Skill Score (FSS):sup:`[1]` and its decompositions
    for a given forecast and observed 2-D field.

    The FSS is computed by counting the squared sum of forecast and observation
    events in a given window size. This is repeated over all possible window
    positions that can fit in the input arrays. A common method to do this is
    via a sliding window.

    While the various compute backends may have their own implementations; most
    of them use some form of prefix sum algorithm to compute this quickly.

    For 2-D fields this data structure is known as the "Summed Area
    Table" :sup:`[2]`.

    Once the squared sums are computed, the final FSS value can be derived by
    accumulating the squared sums :sup:`[3]`.


    The caller is responsible for making sure the input fields are in the 2-D
    spatial domain. (Although it should work for any ``np.array`` as long as it's
    2D, and ``window_size`` is appropriately sized.)

    Args:
        fcst: An array of forecasts
        obs: An array of observations (same spatial shape as ``fcst``)
        event_threshold: A scalar to compare ``fcst`` and ``obs`` fields to generate a
            binary "event" field.
        window_size: A pair of positive integers ``height, width)`` of the sliding
            window; the window dimensions must be greater than 0 and fit within
            the shape of ``obs`` and ``fcst``.
        padding: To handle the edge points, user can use either zero or reflective padding.
            If set to ``"zero"``, applies a 0-valued window border around the
            data field before computing the FSS. If set to ``"reflective"``, the values at the
            edges are mirrored outward. If set to None (default), it uses the edges
            (thickness = window size) of the input fields as the border.
            One can think of it as:
            - padding = None => inner border
            - padding = "zero" => outer border with 0 values
            - padding = "reflective" => outer border with mirrored values
        threshold_operator: The threshold operator used to generate the binary
            event field. E.g. ``np.greater``. Note: this may depend on the backend
            ``compute method`` and not all operators may be supported. Generally,
            ``np.greater``, ``np.less`` and their counterparts. Defaults to "np.greater".
        compute_method: currently only supports ``FssComputeMethod.NUMPY``
            see: :py:class:``FssComputeMethod``

    Returns:
        A float representing the accumulated FSS.

    Raises:
        DimensionError: Various errors are thrown if the input dimensions
            do not conform. e.g. if the window size is larger than the input
            arrays, or if the the spatial dimensions of the input arrays do
            not match.

    References:
        1. Roberts, N. M., and H. W. Lean, 2008: Scale-Selective Verification of Rainfall
           Accumulations from High-Resolution Forecasts of Convective Events. Monthly Weather Review
           136, 78–97, https://doi.org/10.1175/2007mwr2123.1.
        2. https://en.wikipedia.org/wiki/Summed-area_table
        3. FAGGIAN, N., B. ROUX, P. STEINLE, and B. EBERT, 2015: Fast calculation of the fractions
           skill score. MAUSAM, 66, 457–466, https://doi.org/10.54302/mausam.v66i3.555.

    Example:
        >>> import numpy as np
        >>> from scores.spatial import fss_2d_single_field
        >>> fcst = np.random.normal(loc=0.0, scale=1.0, size = (100, 100))
        >>> obs = np.random.normal(loc=1.0, scale=1.0, size = (100, 100))
        >>> fss = fss_2d_single_field(fcst, obs, event_threshold=0.1, window_size=(21, 921))
    """
    np_thrsh_op = _make_numpy_threshold_operator(threshold_operator)

    fss_backend = get_compute_backend(compute_method)

    fb_obj = fss_backend(
        fcst,
        obs,
        window_size=window_size,
        padding=padding,
        event_threshold=event_threshold,
        threshold_operator=np_thrsh_op,
    )

    fss_score, fbs, fbs_ref = fb_obj.compute_fss()

    return {"FSS": fss_score, "FBS": fbs, "FBS_ref": fbs_ref}  # type: ignore


def _make_numpy_threshold_operator(threshold_operator: Optional[Callable]) -> NumpyThresholdOperator:
    thrsh_op: Callable

    if threshold_operator is None:
        thrsh_op = np.greater
    else:
        thrsh_op = threshold_operator

    return NumpyThresholdOperator(thrsh_op)


def _aggregate_fss_decomposed(fss_d: FssDecomposed) -> Dict[str, np.float64]:
    """
    Aggregates the results of decomposed fss scores.
    """
    # can't do ufuncs over custom void types currently...
    fcst_sum = 0.0
    obs_sum = 0.0
    diff_sum = 0.0

    if isinstance(fss_d, np.ndarray):
        l = fss_d.size

        if l < 1:
            return np.float64(0.0)

        with np.nditer(fss_d) as it:
            for elem in it:
                (fcst_, obs_, diff_) = elem.item()  # type: ignore  # mypy is confused
                fcst_sum += fcst_ / l
                obs_sum += obs_ / l
                diff_sum += diff_ / l
    else:
        (fcst_sum, obs_sum, diff_sum) = fss_d

    fss = 0.0
    denom = obs_sum + fcst_sum

    if denom > 0.0:
        fss = 1.0 - diff_sum / denom

    fss_clamped = max(min(fss, 1.0), 0.0)

    return {"fss": np.float64(fss_clamped), "fbs": np.float64(diff_sum), "fbs_ref": np.float64(denom)}  # type: ignore
