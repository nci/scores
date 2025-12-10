import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage
import xarray as xr
from scipy.optimize import minimize

from scores.continuous.standard_impl import mse, rmse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _generate_largest_rain_area_2d(
    fcst: xr.DataArray, obs: xr.DataArray, threshold: float, min_points: int
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    Identify and extract the largest contiguous rain blobs from forecast and observation fields.

    This function expects 2-D spatial inputs and operates like an image segmentation on a
    horizontal grid.
    It masks values below a given threshold and labels connected components (blobs)
    in the forecast and observation arrays. It retains only the largest blob from each field,
    where "largest" refers to the blob with the greatest number of grid points (i.e., pixel count)

    Args:
        fcst (xr.DataArray): 2-D forecast field.
        obs (xr.DataArray): 2-D observation field.
        threshold (float): Minimum value that a grid point must meet or exceed to be considered
            part of a rain blob.
        min_points (int): Minimum number of grid points required for a blob to be ratined

    Returns:
        Largest contiguous blobs from forecast and observation.

    Example:
        >>> fcst_blob, obs_blob = _generate_largest_rain_area_2d(fcst, obs, threshold=5.0, min_points=10)
    """

    masked_obs = obs.where(obs >= threshold)
    masked_fcst = fcst.where(fcst >= threshold)

    # If fcst/obs don't meet minimum counts even pre-blobification
    if masked_fcst.count() < min_points or masked_obs.count() < min_points:
        logger.info(f"Less than {min_points} points meet the condition.")
        masked_fcst[:] = np.nan
        masked_obs[:] = np.nan

        return masked_fcst, masked_obs

    # Label connected components in the masked array
    # This is to identify and group connected regions (or blobs) in an array
    # Connected components are groups of adjacent elements in an array that share the same value
    # in our case, non-NaN values

    # Connectivity: how elements are considered connected
    # Define connectivity for the labeling. 3x3 => 8-connected in 2D
    # (includes N, NE, E, SE, S, SW, W, W, NW neighbours)
    structure = np.ones((3, 3))

    # Assign a unique label to each connected component. For instance, if there are 3 separate
    # blobs in our array, each blob will be assigned a different label (e.g., 1, 2, 3)
    labeled_array_obs, num_features_obs = scipy.ndimage.label(
        ~np.isnan(masked_obs), structure=structure
    )  # labels the connected components in the masked array where the values are not NaN
    if num_features_obs > 1:
        # Find the largest blob
        largest_blob_label_obs = np.argmax(np.bincount(labeled_array_obs.flat)[1:]) + 1

        # Create a new masked array with only the largest blob
        obs = masked_obs.where(labeled_array_obs == largest_blob_label_obs)
    else:
        obs = masked_obs

    labeled_array_fcst, num_features_fcst = scipy.ndimage.label(
        ~np.isnan(masked_fcst), structure=structure
    )  # labels the connected components in the masked array where the values are not NaN
    if num_features_fcst > 1:
        # Find the largest blob
        largest_blob_label_fcst = np.argmax(np.bincount(labeled_array_fcst.flat)[1:]) + 1

        # Create a new masked array with only the largest blob
        fcst = masked_fcst.where(labeled_array_fcst == largest_blob_label_fcst)
    else:
        fcst = masked_fcst

    # Retain only the largest contiguous blob in each field; all other values are set to NaN
    fcst_blob = fcst
    obs_blob = obs

    # Apply min_points check to the extracted blobs
    if fcst_blob.count() < min_points or obs_blob.count() < min_points:
        logger.info(f"Largest blob has fewer than {min_points} points.")
        fcst_blob[:] = np.nan
        obs_blob[:] = np.nan

    return fcst_blob, obs_blob


def _calc_bounding_box_centre(data_array: xr.DataArray) -> Tuple[int, int]:
    """
    Compute the centre of the bounding box for valid (non-NaN and non-zero) values in a 2D data array.
    This function assumes the input is a 2D field and is intended for use with a single contiguous
    region (blob). It computes the geometric centre of the bounding box enclosing all valid points.
    Args:
        data_array (xr.DataArray): Input 2D data array.

    Returns:
        (row_index, column_index) of the bounding box centre in array index space.

    Example:
        >>> centre = _calc_bounding_box_centre(data)
    """

    # Convert to NumPy array and mask NaNs
    masked_array = np.ma.masked_invalid(data_array.values)

    # Get indices of valid (non-NaN) and non-zero values
    valid_indices = np.argwhere(masked_array > 0)

    if valid_indices.size == 0:
        return (np.nan, np.nan)

    # Compute bounding box from array indices
    min_y, min_x = valid_indices.min(axis=0)
    max_y, max_x = valid_indices.max(axis=0)

    # Compute centre of bounding box in index space (not coordinate space)
    centre_y = int((min_y + max_y) / 2)
    centre_x = int((min_x + max_x) / 2)

    return (centre_y, centre_x)


def _translate_forecast_region(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    y_name: str,
    x_name: str,
    max_distance: float,
    coord_units: str,
) -> Tuple[xr.DataArray, int, int]:
    """
    Translate the forecast field to best spatially align with the observation field.

    This function performs a 2D spatial translation (no rotation or scaling)
    to minimize the MSE between forecast and observation.

    Args:
        fcst (xr.DataArray): Forecast field.
        obs (xr.DataArray): Observation field.
        x_name (str): Name of the zonal spatial dimension (e.g., 'x' or 'longitude').
        y_name (str): Name of the meridional spatial dimension (e.g., 'y' or 'latitude').
        max_distance (float) : Maximum distance in km allowed for the shifted blob.
        coord_units (str) : coordinates units, 'degrees' or 'metres'


    Returns:
        Translated forecast and optimal shift values in grid points (dx, dy).

    Example:
        >>> shifted_fcst, delta_x, delta_y = _translate_forecast_region(fcst, obs, 'y', 'x', 300, 'metres')
    """

    # Create fixed mask based on observation availability
    # Ensure that no matter where the forecast is shifted,
    # the evaluation is always done over the same observation-valid region
    fixed_mask = ~np.isnan(obs)

    # Mask forecast and observation using fixed mask
    fcst_masked = fcst.where(fixed_mask)
    obs_masked = obs.where(fixed_mask)

    original_mse = float(mse(fcst_masked, obs_masked))

    # Brute-force search
    best_score = np.inf
    best_shift = None
    shift_range = range(-10, 11)
    for dy in shift_range:
        for dx in shift_range:
            shift = [dx, dy]
            mse_score = _shifted_mse(shift, fcst, obs, [y_name, x_name], fixed_mask)
            if np.isfinite(mse_score) and mse_score < best_score:
                best_score = mse_score
                best_shift = shift

    # Refine with local optimization from brute-force result
    result = minimize(
        _shifted_mse,
        best_shift,
        args=(fcst, obs, [y_name, x_name], fixed_mask),
        method="Nelder-Mead",
    )
    optimal_shift = result.x if result.success and np.isfinite(result.fun) else None

    # Fallback to bounding box centre if optimization fails
    if optimal_shift is None:
        logger.info("Optimization failed. Falling back to bounding box centre alignment.")
        fcst_bounding_box_centre = _calc_bounding_box_centre(fcst)  # [y,x]
        obs_bounding_box_centre = _calc_bounding_box_centre(obs)
        optimal_shift = [
            obs_bounding_box_centre[1] - fcst_bounding_box_centre[1],  # x_shift
            obs_bounding_box_centre[0] - fcst_bounding_box_centre[0],  # y_shift
        ]

    # Apply shift
    dx, dy = np.round(optimal_shift[0]), np.round(optimal_shift[1])

    # Compute shift distance in km
    resolution_km = _calc_resolution(obs, [y_name, x_name], coord_units)

    shift_distance_km = resolution_km * np.sqrt(dx**2 + dy**2)

    if shift_distance_km > max_distance:
        logger.info(f"Rejected shift: {shift_distance_km:.2f} km > {max_distance} km")
        # What does it mean when this happens? Should the overall metric be done,
        # or just the shift not occur?
        return None, None, None

    shifted_fcst = _shift_fcst(fcst, shift_x=dx, shift_y=dy, spatial_dims=[y_name, x_name])

    # Final evaluation using fixed mask
    shifted_fcst_masked = shifted_fcst.where(fixed_mask)
    mse_shifted = float(mse(shifted_fcst_masked, obs_masked))
    corr_shifted = _calc_corr_coeff(shifted_fcst_masked, obs_masked)
    rmse_shifted = float(rmse(shifted_fcst_masked, obs_masked))

    rmse_original = float(rmse(fcst_masked, obs_masked))
    corr_original = _calc_corr_coeff(fcst_masked, obs_masked)

    # How can the following actually occur? What should users do about it? What does it mean?
    # If we don't know, should we warn instead?
    # If not, should we just let the user decide?

    # if (
    #     rmse_shifted > rmse_original
    #     or corr_shifted < corr_original
    #     or mse_shifted > original_mse
    # ):
    #     return None, None, None

    return shifted_fcst, dx, dy


def nansafe_int(value):

    if np.isnan(value):
        return value

    return int(value)


def _shift_fcst(fcst: xr.DataArray, shift_x: int, shift_y: int, spatial_dims: List[str]) -> xr.DataArray:
    """
    Apply a spatial shift to a 2D forecast field along specified spatial dimensions.

    This function assumes a 2D spatial field and shifts it along the provided
    spatial dimensions.

    Args:
        fcst (xr.DataArray): 2D forecast field.
        shift_x (int): Shift along x-dimension.
        shift_y (int): Shift along y-dimension.
        spatial_dims (list[str]):Names of the 2 spatial dimensions, ordered as [y_dim, x_dim]

    Returns:
        Forecast field shifted spatially.

    Example:
        >>> shifted = _shift_fcst(fcst, 2, -1, ['y', 'x'])
    """
    # Unpack spatial dimension names
    ydim, xdim = spatial_dims

    # Define shift amounts for each dim
    shift_xdim = nansafe_int(shift_x)  # dx => X dim
    shift_ydim = nansafe_int(shift_y)  # dy => Y dim

    shift_xdim = 0 if np.isnan(shift_xdim) else shift_xdim
    shift_ydim = 0 if np.isnan(shift_ydim) else shift_ydim

    shifts_kwargs = {
        xdim: shift_xdim,
        ydim: shift_ydim,
    }

    shifted = fcst.shift(
        **shifts_kwargs,
        fill_value=np.nan,
    )

    return shifted


def _shifted_mse(
    shifts: List[int],
    fcst: xr.DataArray,
    obs: xr.DataArray,
    spatial_dims: List[str],
    fixed_mask: xr.DataArray,
) -> float:
    """
    Objective function for optimization: computes MSE between shifted forecast and observation.

    Args:
        shifts (List[int]): Shift values [x, y].
        fcst (xr.DataArray): Forecast field.
        obs (xr.DataArray): Observation field.
        spatial_dims (List[str]): List of spatial dimension names.

    Returns:
        MSE value for the given shift.

    Example:
        >>> error = _shifted_mse([1, -2], fcst, obs, ['x', 'y'])
    """
    # Validate input
    if len(shifts) != 2 or np.any(np.isnan(shifts)):
        return np.inf

    try:
        shift_x, shift_y = int(round(shifts[0])), int(round(shifts[1]))
    except (ValueError, TypeError):
        return np.inf

    shifted_fcst = _shift_fcst(fcst, shift_x, shift_y, spatial_dims)

    # Mask forecast using fixed obs mask
    fcst_masked = shifted_fcst.where(fixed_mask)
    obs_masked = obs.where(fixed_mask)

    # To avoid artefacts, we allow shifts as long as the main forecast blob
    # (at least 80%) stays within the valid observation region
    valid_fraction = np.sum(~np.isnan(fcst_masked)) / np.sum(~np.isnan(fcst))

    if valid_fraction < 0.8:
        return np.inf

    mse_val = float(mse(fcst_masked, obs_masked))
    corr_val = _calc_corr_coeff(fcst_masked, obs_masked)

    # Penalize low correlation
    penalty = 1e3 if np.isnan(corr_val) or corr_val < 0.3 else 0

    return mse_val + penalty


def _calc_mse_volume(shifted_fcst: xr.DataArray, obs: xr.DataArray) -> float:
    """
    Calculate the volume error component of CRA score.

    $$\text{Volume Error} = (\bar{f} - \bar{o})^2$$

    where $\bar{f}$ and $\bar{o}$ are the mean values of forecast and observation respectively.

    Args:
        shifted_fcst (xr.DataArray): Shifted forecast field.
        obs (xr.DataArray): Observation field.

    Returns:
        Volume error.

    Example:
        >>> volume_error = _calc_mse_volume(shifted_fcst, obs)
    """
    # mean_shifted_forecast = float(shifted_fcst.mean(skipna=True).values)
    mean_shifted_forecast = np.mean(shifted_fcst)
    mean_observed = np.mean(obs)
    volume_error = (mean_shifted_forecast - mean_observed) ** 2
    return volume_error


def _calc_num_points(data: xr.DataArray, threshold: float) -> int:
    """
    Count the number of grid points in the data above a given threshold.

    Args:
        data (xr.DataArray): Input data array.
        threshold (float): Threshold value.

    Returns:
        Number of points above the threshold.

    Example:
        >>> count = _calc_num_points(data, threshold=5.0)
    """
    mask = data >= threshold
    count_above_threshold = mask.sum().item()
    return count_above_threshold


def _calc_corr_coeff(data1: xr.DataArray, data2: xr.DataArray) -> float:
    """
    Calculate the Pearson correlation coefficient between two data arrays.

    Args:
        data1 (xr.DataArray): First data array.
        data2 (xr.DataArray): Second data array.

    Returns:
        Correlation coefficient.

    Example:
        >>> corr = _calc_corr_coeff(data1, data2)
    """

    data1_flat = data1.values.flatten()
    data2_flat = data2.values.flatten()

    # Remove NaNs
    mask = ~np.isnan(data1_flat) & ~np.isnan(data2_flat)
    data1_clean = data1_flat[mask]
    data2_clean = data2_flat[mask]

    # Check for empty or constant arrays
    if len(data1_clean) == 0 or len(data2_clean) == 0:
        return np.nan
    if np.all(data1_clean == data1_clean[0]) or np.all(data2_clean == data2_clean[0]):
        return np.nan
    cc = np.corrcoef(data1_clean, data2_clean)[0, 1]
    return float(cc)


def _calc_resolution(obs: xr.DataArray, spatial_dims: list[str], units: str) -> float:
    """
    Compute average grid resolution in kilometres.

    Args:
        obs (xr.DataArray): Input data array.
        spatial_dims (list[str]): Names of spatial dimensions [y_dim, x_dim].
        units (str): Units of coordinates. Must be 'degrees' or 'metres'.

    Returns:
        float: Average resolution in kilometres.
    """
    y_coords = obs.coords[spatial_dims[0]].values
    x_coords = obs.coords[spatial_dims[1]].values

    # Compute resolution in native units
    dy = np.mean(np.diff(y_coords))
    dx = np.mean(np.diff(x_coords))

    if units == "degrees":
        lat_mean = np.mean(y_coords)
        dy_km = np.abs(dy) * 111
        dx_km = np.abs(dx) * 111 * np.cos(np.radians(lat_mean))
    elif units == "metres":
        dy_km = np.abs(dy) / 1000
        dx_km = np.abs(dx) / 1000
    else:
        # Should not be reachable due to earlier validation step
        raise ValueError("units must be 'degrees' or 'metres'")

    avg_resolution_km = np.mean([dy_km, dx_km])
    return float(avg_resolution_km)


def _cra_image(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
    y_name: str,
    x_name: str,
    max_distance: float = 300,
    min_points: int = 10,
    coord_units: str = "metres",
    extra_components: bool = False,
    time_name: Optional[str] = None,
) -> xr.Dataset:
    """
        Compute the Contiguous Rain Area (CRA) score between forecast and observation fields.
        This function is designed for 2D spatial fields. For time-dependent data,
        use the :py:func:`scores.spatial.cra` function.
        For ensemble data, apply this function to each realization individually or compute
        the ensemble mean beforehand.

        The CRA score decomposes the total mean squared error (MSE) into three components:
        displacement, volume, and pattern. It identifies contiguous rain blobs above a threshold,
        shifts the forecast blob to best match the observed blob, and evaluates the error reduction.

        The decomposition is defined as:

        .. math::
            \\text{MSE}_{\\text{total}} = \\text{MSE}_{\\text{displacement}} + \\text{MSE}_{\\text{pattern}} + \\text{MSE}_{\\text{volume}}

        where:
            - ``MSE_displacement`` is the error reduction due to optimal spatial alignment,
            - ``MSE_volume`` is the error due to differences in blob intensity,
            - ``MSE_pattern`` is the residual error after alignment and volume adjustment.

        This metric is spatial and blob-based, and does not support ``reduce_dims``, ``preserve_dims``, or ``weights``.

        Args:
            fcst (xr.DataArray): Forecast field as an xarray DataArray.
            obs (xr.DataArray): Observation field as an xarray DataArray.
            threshold (float): Threshold to define contiguous rain areas.
            y_name (str): Name of the meridional spatial dimension (e.g., 'lat', 'projection_y_coordinate').
            x_name (str): Name of the zonal spatial dimension (e.g., 'lon', 'projection_x_coordinate').
            max_distance (float): Maximum allowed translation distance in kilometres.
            min_points (int): Minimum number of grid points required in a blob.
            coord_units (str) : Coordinate units, 'degrees' or 'metres'

        Returns:
            `CRA2DMetric`: A dictionary containing the CRA components and diagnostics.

            Returns an object containing NaNs if input data is invalid or CRA computation fails.

        Raises:
            ValueError: If input shapes do not match or blobs cannot be computed.
            TypeError: If inputs are not xarray DataArrays.

        References:
            Ebert, E. E., & McBride, J. L. (2000). Verification of precipitation forecasts from operational numerical weather prediction models. *Weather and Forecasting*, 15(3), 247-263. https://doi.org/10.1016/S0022-1694(00)00343-7
            Ebert, E. E., and W. A. Gallus , 2009: Toward Better Understanding of the Contiguous Rain Area (CRA) Method for Spatial Forecast Verification. *Weather and Forecasting*, 24, 1401-1415, https://doi.org/10.1175/200
    9WAF2222252.1

        Example:
            >>> from scores.spatial import cra
            >>> import xarray as xr
            >>> fcst = xr.DataArray(...)  # your forecast data
            >>> obs = xr.DataArray(...)   # your observation data
            >>> result = cra(fcst, obs, threshold=5.0)
            >>> print(result['mse_total'])

    """

    # Throw an exception if invalid input
    validate_cra2d_inputs(fcst, obs, time_name, coord_units, x_name, y_name)

    fcst_blob, obs_blob = _generate_largest_rain_area_2d(fcst, obs, threshold, min_points)

    mse_total = mse(fcst_blob, obs_blob)

    shifted_fcst, delta_x, delta_y = _translate_forecast_region(
        fcst_blob, obs_blob, y_name, x_name, max_distance, coord_units
    )
    optimal_shift = [delta_x, delta_y]

    mse_shift = mse(shifted_fcst, obs_blob)
    mse_displacement = mse_total - mse_shift
    mse_volume = _calc_mse_volume(shifted_fcst, obs_blob)
    mse_pattern = mse_shift - mse_volume

    num_gridpoints_above_threshold_fcst = _calc_num_points(fcst_blob, threshold)
    num_gridpoints_above_threshold_obs = _calc_num_points(obs_blob, threshold)

    data_vars = {
        "mse_total": mse_total,
        "mse_shift": mse_shift,
        "mse_displacement": mse_total - mse_shift,
        "mse_volume": mse_volume,
        "mse_pattern": mse_shift - mse_volume,
    }

    if extra_components:
        extra_vars = {
            "fcst_blob": fcst_blob,
            "obs_blob": obs_blob,
            "shifted_fcst": shifted_fcst,
            "optimal_shift": optimal_shift,
            "num_gridpoints_above_threshold_fcst": num_gridpoints_above_threshold_fcst,
            "num_gridpoints_above_threshold_obs": num_gridpoints_above_threshold_obs,
            "avg_fcst": np.mean(fcst_blob),
            "avg_obs": np.mean(obs_blob),
            "max_fcst": np.max(fcst_blob),
            "max_obs": np.max(obs_blob),
            "corr_coeff_original": _calc_corr_coeff(fcst_blob, obs_blob),
            "corr_coeff_shifted": _calc_corr_coeff(shifted_fcst, obs_blob),
            "rmse_original": rmse(fcst_blob, obs_blob),
            "rmse_shifted": rmse(shifted_fcst, obs_blob),
        }

        data_vars = data_vars | extra_vars

    coords = [x_name, y_name]
    if time_name:
        coords = [time, x_name, y_name]

    ds = xr.Dataset(coords={name: obs[name] for name in coords}, data_vars=data_vars)

    return ds


def cra(
    fcst: XarrayLike,
    obs: XarrayLike,
    threshold: float,
    y_name: str,
    x_name: str,
    max_distance: float = 300,
    min_points: int = 10,
    coord_units: str = "metres",
    extra_components: bool = False,
) -> xr.Dataset:
    """
        Compute Contiguous Rain Area (CRA) metrics across grouped slices of a forecast and
        observation field.

        This function extends :py:func:`scores.spatial.cra_image` to handle time series.

        It applies CRA decomposition to each slice along the specified ``reduce_dims`` and
        aggregates results into lists.

        CRA decomposes the total mean squared error (MSE) into:
            - Displacement: Error reduction due to optimal spatial alignment.
            - Volume: Error due to intensity differences.
            - Pattern: Residual error after alignment and volume adjustment.

        For each time, the algorithm:
            1. Identifies contiguous rain blobs above ``threshold``.
            2. Computes optimal spatial shift within ``max_distance``.
            3. Calculates CRA components and diagnostics.

        .. math::
            \\text{MSE}_{\\text{total}} = \\text{MSE}_{\\text{displacement}} + \\text{MSE}_{\\text{pattern}} + \\text{MSE}_{\\text{volume}}

        Args:
            fcst (xr.DataArray): Forecast field with at least one grouping dimension (e.g., time).
            obs (xr.DataArray): Observation field aligned with ``fcst``.
            threshold (float): Threshold to define contiguous rain areas.
            y_name (str): Name of the meridional spatial dimension (e.g., 'lat', 'projection_y_coordinate').
            x_name (str): Name of the zonal spatial dimension (e.g., 'lon', 'projection_x_coordinate').
            max_distance (float): Maximum allowed translation distance in kilometres.
            min_points (int): Minimum number of grid points required in a blob.
            reduce_dims (list[str] or str, optional): Dimension to group by (default: ["time"]).
            coord_units (str) : Coordinate units, 'degrees' or 'metres'

        Returns:
            A dictionary where each key corresponds to a CRA metric and maps to a list of
            values, one for each slice along the specified grouping dimension (e.g. time):
                - mse_total (list[float]): Total mean squared error between forecast and observed blobs.
                - mse_displacement (list[float]): MSE due to spatial displacement between forecast and observed blobs.
                - mse_volume (list[float]): MSE due to volume differences.
                - mse_pattern (list[float]): MSE due to pattern/structure differences.
                - optimal_shift (list[list[int]]): Optimal [x, y] shift applied to forecast blob in grid points units.
                - num_gridpoints_above_threshold_fcst (list[int]): Number of grid points in forecast blob above threshold.
                - num_gridpoints_above_threshold_obs (list[int]): Number of grid points in observed blob above threshold.
                - avg_fcst (list[float]): Mean value of the forecast blob.
                - avg_obs (list[float]): Mean value of the observed blob.
                - max_fcst (list[float]): Maximum value in the forecast blob.
                - max_obs (list[float]): Maximum value in the observed blob.
                - corr_coeff_original (list[float]): Correlation coefficient between original forecast and observed blobs.
                - corr_coeff_shifted (list[float]): Correlation coefficient between shifted forecast and observed blobs.
                - rmse_original (list[float]): Root mean square error between original forecast and observed blobs.
                - rmse_shifted (list[float]): Root mean square error between shifted forecast and observed blobs.
            Returns None if input data is invalid or CRA computation fails.


        Raises:
            ValueError: If input shapes do not match or grouping dimension is invalid.
            TypeError: If inputs are not xarray DataArrays.

        References:
            Ebert, E. E., & McBride, J. L. (2000). Verification of precipitation forecasts from operational numerical weather prediction models. *Weather and Forecasting*, 15(3), 247-263. https://doi.org/10.1016/S0022-1694(00)00343-7
            Ebert, E. E., and W. A. Gallus , 2009: Toward Better Understanding of the Contiguous Rain Area (CRA) Method for Spatial Forecast Verification. *Weather and Forecasting*, 24, 1401-1415, https://doi.org/10.1175/200
    9WAF2222252.1

        Example:
            >>> from scores.spatial import cra
            >>> import xarray as xr
            >>> fcst = xr.DataArray(...)  # forecast with time dimension
            >>> obs = xr.DataArray(...)   # observation with time dimension
            >>> result = cra(fcst, obs, threshold=5.0, reduce_dims="time")
            >>> print(result["mse_total"])
    """

    # --- Input validation ---
    if not isinstance(fcst, xr.DataArray):
        raise TypeError("fcst must be an xarray DataArray")
    if not isinstance(obs, xr.DataArray):
        raise TypeError("obs must be an xarray DataArray")

    if fcst.shape != obs.shape:
        raise ValueError("fcst and obs must have the same shape")

    # Align forecast and observation
    fcst, obs = xr.align(fcst, obs)

    extra_dims = [d for d in fcst.dims if d not in [x_name, y_name]]

    if extra_dims:
        stacked_fcst = fcst.stack({"stacked": extra_dims})
        stacked_obs = obs.stack({"stacked": extra_dims})
        results = []

        for i in range(len(stacked_fcst.stacked)):
            fcst_image = stacked_fcst.isel(stacked=i)
            obs_image = stacked_obs.isel(stacked=i)
            r = _cra_image(fcst_image, obs_image, threshold, y_name, x_name, extra_components=extra_components)
            results.append(r)

        if len(results) == 1:
            result = results[0]

        else:
            result = xr.concat(results, dim="stacked").set_index(stacked=extra_dims)
            result = result.unstack()

    else:
        result = _cra_image(fcst, obs, threshold, y_name, x_name, extra_components=extra_components)
    return result


def validate_cra2d_inputs(fcst, obs, time_name, coord_units, x_name, y_name):

    if not isinstance(fcst, xr.DataArray):
        raise TypeError("fcst must be an xarray DataArray")
    if not isinstance(obs, xr.DataArray):
        raise TypeError("obs must be an xarray DataArray")
    if fcst.shape != obs.shape:
        raise ValueError("fcst and obs must have the same shape")

    max_allowed_coords = 2
    if time_name:
        max_allowed_coords = 3
        if len(fcst[time_name] != 1):
            raise ValueError("The time dimension can only have a length of one (single sample) in the 2d score")

    if len(fcst.shape) != max_allowed_coords:
        raise ValueError("The `fcst` inputs contain additional coordinate dimensions which cannot be handled")

    if len(obs.shape) != max_allowed_coords:
        raise ValueError("The `obs` inputs contain additional coordinate dimensions which cannot be handled")

    if fcst.shape != obs.shape:
        raise ValueError("fcst and obs must have the same shape")

    for dim in [y_name, x_name]:
        if dim not in obs.dims:
            raise ValueError(f"Spatial dimension '{dim}' not found in observation data")

        if dim not in fcst.dims:
            raise ValueError(f"Spatial dimension '{dim}' not found in forecast data")

    allowed_units = ["degrees", "metres"]
    if coord_units not in allowed_units:
        raise ValueError(f"coord_units must be one of {allowed_units}")


# TODO: Merge the docs with cra_image and delete this method
def def_core_2d(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
    y_name: str,
    x_name: str,
    max_distance: float = 300,
    min_points: int = 10,
    coord_units: str = "metres",
    time_name: Optional[str] = None,  # Specify a length-of-one time dimension name
) -> Optional[dict]:
    """
    Compute the core Contiguous Rain Area (CRA) decomposition between forecast and observation fields.

    This function returns only the essential CRA decomposition components.
    To obtain all CRA metrics and diagnostics, use :py:func:`scores.spatial.cra_image`. For time-dependent data,
    use the :py:func:`scores.spatial.cra` function.

    The core CRA score decomposes the total mean squared error (MSE) into three components:
    displacement, volume, and pattern. It identifies contiguous rain blobs above a threshold,
    shifts the forecast blob to best match the observed blob, and evaluates the error reduction.

    See :py:func:`scores.spatial.cra_image` for more details.

    Args:
        fcst (xr.DataArray): Forecast field as an xarray DataArray.
        obs (xr.DataArray): Observation field as an xarray DataArray.
        threshold (float): Threshold to define contiguous rain areas.
        y_name (str): Name of the meridional spatial dimension
            (e.g., 'lat', 'projection_y_coordinate').
        x_name (str): Name of the zonal spatial dimension
            (e.g., 'lon', 'projection_x_coordinate').
        time_name (Optional[str]): Name of the dimension to use for time-series (e.g. 'time', 'lead_time' or 'valid_time')
        max_distance (float): Maximum allowed translation distance in kilometres.
        min_points (int): Minimum number of grid points required in a blob.
        coord_units (str) : Coordinate units, 'degrees' or 'metres'

    Returns:
        A CRAMetric instance containing the core CRA components
            - mse_total (float): Total mean squared error between forecast and observed blobs.
            - mse_displacement (float): MSE reduction due to spatial displacement after optimal alignment.
            - mse_volume (float): MSE due to mean intensity (volume) differences.
            - mse_pattern (float): Residual MSE after displacement and volume adjustment.
            - optimal_shift (list[int]): Optimal [x, y] shift applied to the forecast blob (grid-point units).


    Example:
        >>> import xarray as xr
        >>> from scores.spatial import cra_image
        >>> fcst = xr.DataArray(...)  # 2D forecast
        >>> obs = xr.DataArray(...)   # 2D observation
        >>> result = cra_image(fcst, obs, threshold=5.0, y_name="lat", x_name="lon")
        >>> print(result.mse_total)


    """

    # Throw an exception if invalid input
    validate_cra2d_inputs(fcst, obs, time_name, coord_units, x_name, y_name)

    fcst_blob, obs_blob = _generate_largest_rain_area_2d(fcst, obs, threshold, min_points)
    mse_total = float(mse(fcst_blob, obs_blob))

    shifted_fcst, dx, dy = _translate_forecast_region(fcst_blob, obs_blob, y_name, x_name, max_distance, coord_units)
    assert shifted_fcst is not None
    assert obs_blob is not None
    mse_shift = mse(shifted_fcst, obs_blob)
    mse_volume = _calc_mse_volume(shifted_fcst, obs_blob)

    data_vars = {
        "mse_total": mse_total,
        "mse_shift": mse_shift,
        "mse_displacement": mse_total - mse_shift,
        "mse_volume": mse_volume,
        "mse_pattern": mse_shift - mse_volume,
    }

    coords = [x_name, y_name]
    if time_name:
        coords = [time, x_name, y_name]

    ds = xr.Dataset(coords={name: obs[name] for name in coords}, data_vars=data_vars)

    return ds
