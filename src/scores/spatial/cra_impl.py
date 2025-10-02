from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage
import xarray as xr
from scipy.optimize import minimize

from scores.continuous.standard_impl import mse, rmse

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_blobs(
    fcst: xr.DataArray, obs: xr.DataArray, threshold: float
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    Identify and extract the largest contiguous rain blobs from forecast and observation fields.

    This function masks values below a given threshold and labels connected components (blobs)
    in the forecast and observation arrays. It retains only the largest blob from each field.

    Args:
        fcst (xr.DataArray): Forecast field.
        obs (xr.DataArray): Observation field.
        threshold (float): Threshold to define rain blobs.

    Returns:
        Tuple[xr.DataArray, xr.DataArray]: Largest contiguous blobs from forecast and observation.

    Example:
        >>> fcst_blob, obs_blob = generate_blobs(fcst, obs, threshold=5.0)
    """

    masked_obs = obs.where(obs > threshold)
    masked_fcst = fcst.where(fcst > threshold)

    if masked_fcst.count() < 10 or masked_obs.count() < 10:
        logger.info("Less than 10 points meet the condition.")
        return None, None

    # Label connected components in the masked array
    # This is to identify and group connected regions (or blobs) in an array
    # Connected components are groups of adjacent elements in an array that share the same value
    # in our case, non-NaN values

    # Connectivity: how elements are considered connected
    structure = np.ones((3, 3))  # Define connectivity for the labeling. 3x3 => 8-connected in 2D

    # Assign a unique label to each connected component. For instance, if there are 3 separate
    # blobs in our array, each blob will be assigned a different label (e.g., 1, 2, 3)
    labeled_array_obs, num_features_obs = scipy.ndimage.label(
        ~np.isnan(masked_obs), structure=structure
    )  # labels the connected components in the masked array where the values are not NaN
    if num_features_obs > 1:
        # Find the largest blob
        largest_blob_label = np.argmax(np.bincount(labeled_array_obs.flat)[1:]) + 1

        # Create a new masked array with only the largest blob
        obs = masked_obs.where(labeled_array_obs == largest_blob_label)
    else:
        obs = masked_obs

    labeled_array_fcst, num_features_fcst = scipy.ndimage.label(
        ~np.isnan(masked_fcst), structure=structure
    )  # labels the connected components in the masked array where the values are not NaN
    if num_features_fcst > 1:
        # Find the largest blob
        largest_blob_label = np.argmax(np.bincount(labeled_array_fcst.flat)[1:]) + 1

        # Create a new masked array with only the largest blob
        fcst = masked_fcst.where(labeled_array_fcst == largest_blob_label)
    else:
        fcst = masked_fcst

    fcst_blob = fcst
    obs_blob = obs
    return fcst_blob, obs_blob


def calc_mse(fcst: xr.DataArray, obs: xr.DataArray) -> float:
    r"""
    Calculate the Mean Squared Error (MSE) between forecast and observation arrays.

    $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2$$

    Args:
        fcst (xr.DataArray): Forecast field.
        obs (xr.DataArray): Observation field.

    Returns:
        float: Mean squared error value.

    Example:
        >>> mse = calc_mse(fcst, obs)
    """
    return float(mse(fcst, obs))


def calc_bounding_box_centre(data_array: xr.DataArray) -> Tuple[float, float]:
    """
    Compute the centre of the bounding box for valid (non-NaN and non-zero) values in a data array.

    Args:
        data_array (xr.DataArray): Input data array.

    Returns:
        Tuple[float, float]: Coordinates of the bounding box centre.

    Example:
        >>> centre = calc_bounding_box_centre(data)
    """

    # Convert to NumPy array and mask NaNs
    masked_array = np.ma.masked_invalid(data_array.values)

    # Get indices of valid (non-NaN) and non-zero values
    valid_indices = np.argwhere(masked_array > 0)

    if valid_indices.size == 0:
        return (np.nan, np.nan)

    # Compute bounding box
    min_y, min_x = valid_indices.min(axis=0)
    max_y, max_x = valid_indices.max(axis=0)

    # Compute centre of bounding box
    centre_y = (min_y + max_y) / 2
    centre_x = (min_x + max_x) / 2

    return (centre_y, centre_x)


def calc_shifted_forecast(
    fcst: xr.DataArray, obs: xr.DataArray, spatial_dims: List[str], max_distance: float
) -> Tuple[xr.DataArray, List[int]]:
    """
    Shift the forecast field to best align with the observation field using optimization.

    This function minimizes the MSE between forecast and observation by shifting the forecast spatially.

    Args:
        fcst (xr.DataArray): Forecast field.
        obs (xr.DataArray): Observation field.
        spatial_dims (list[str]): List of spatial dimension names.

    Returns:
        Tuple[xr.DataArray, list[int]]: Shifted forecast and optimal shift values in grid points.

    Example:
        >>> shifted_fcst, shift = calc_shifted_forecast(fcst, obs, ['x', 'y'])
    """

    # Create fixed mask based on observation availability
    # Ensure that no matter where the forecast is shifted,
    # the evaluation is always done over the same observation-valid region
    fixed_mask = ~np.isnan(obs)

    if not fixed_mask.any():
        logger.info("No valid observation data.")
        return None, None

    # Mask forecast and observation using fixed mask
    fcst_masked = fcst.where(fixed_mask)
    obs_masked = obs.where(fixed_mask)

    original_mse = calc_mse(fcst_masked, obs_masked)

    # Brute-force search
    best_score = np.inf
    best_shift = None
    shift_range = range(-10,11)
    for dy in shift_range:
        for dx in shift_range:
            shift = [dx,dy]
            mse_score = objective_function(shift, fcst, obs, spatial_dims, fixed_mask)
            if np.isfinite(mse_score) and mse_score < best_score:
                best_score = mse_score
                best_shift = shift

    # Refine with local optimization from brute-force result
    result = minimize(
        objective_function,
        best_shift,
        args=(fcst, obs, spatial_dims, fixed_mask),
        method="Nelder-Mead"
    )
    optimal_shift = result.x if result.success and np.isfinite(result.fun) else None

    # Fallback to bounding box centre if optimization fails
    if optimal_shift is None:
        logger.info("Optimization failed. Falling back to bounding box centre alignment.")
        fcst_bounding_box_centre = calc_bounding_box_centre(fcst)  # [y,x]
        obs_bounding_box_centre = calc_bounding_box_centre(obs)
        optimal_shift = [
            obs_bounding_box_centre[1] - fcst_bounding_box_centre[1],  # x_shift
            obs_bounding_box_centre[0] - fcst_bounding_box_centre[0],  # y_shift
        ]

    # Apply shift
    dx, dy = int(round(optimal_shift[0])), int(round(optimal_shift[1]))
    
    # Compute shift distance in km
    resolution_km = calc_resolution(obs, spatial_dims)
    shift_distance_km = resolution_km * np.sqrt(dx**2 + dy**2)

    if shift_distance_km > max_distance:
        logger.info(f"Rejected shift: {shift_distance_km:.2f} km > {max_distance} km")
        return None, None

    shifted_fcst = shift_fcst(fcst,shift_x=dx, shift_y=dy, spatial_dims=spatial_dims)


    # Final evaluation using fixed mask
    shifted_fcst_masked = shifted_fcst.where(fixed_mask)
    mse_shifted = calc_mse(shifted_fcst_masked, obs_masked)
    corr_shifted = calc_corr_coeff(shifted_fcst_masked, obs_masked)
    rmse_shifted = calc_rmse(shifted_fcst_masked, obs_masked)

    rmse_original = calc_rmse(fcst_masked, obs_masked)
    corr_original = calc_corr_coeff(fcst_masked, obs_masked)

    if (
        rmse_shifted > rmse_original or
        corr_shifted < corr_original or
        mse_shifted > original_mse
    ):
        return None, None

    return shifted_fcst, [dx, dy]


def calc_rmse(fcst: xr.DataArray, obs: xr.DataArray) -> float:
    r"""
    Calculate the Root Mean Squared Error (RMSE) between forecast and observation arrays.

    $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2}$$

    Args:
        fcst (xr.DataArray): Forecast field.
        obs (xr.DataArray): Observation field.

    Returns:
        float: Root mean squared error value.

    Example:
        >>> rmse = calc_rmse(fcst, obs)
    """
    return float(rmse(fcst, obs))


def shift_fcst(fcst: xr.DataArray, shift_x: int, shift_y: int, spatial_dims: List[str]) -> xr.DataArray:
    """
    Apply a spatial shift to the forecast field.

    Args:
        fcst (xr.DataArray): Forecast field.
        shift_x (int): Shift along x-dimension.
        shift_y (int): Shift along y-dimension.
        spatial_dims (list[str]): List of spatial dimension names.

    Returns:
        xr.DataArray: Shifted forecast field.

    Example:
        >>> shifted = shift_fcst(fcst, 2, -1, ['y', 'x'])
    """
    ydim, xdim = spatial_dims
    shift_kwargs = {
        ydim: int(shift_y),  # dy => Y dim
        xdim: int(shift_x),  # dx => X dim
    }
    return fcst.shift(
        **shift_kwargs,
        fill_value=np.nan,
    )


def objective_function(shifts: List[int], fcst: xr.DataArray, obs: xr.DataArray, spatial_dims: List[str], fixed_mask: xr.DataArray) -> float:
    """
    Objective function for optimization: computes MSE between shifted forecast and observation.

    Args:
        shifts (List[int]): Shift values [x, y].
        fcst (xr.DataArray): Forecast field.
        obs (xr.DataArray): Observation field.
        spatial_dims (List[str]): List of spatial dimension names.

    Returns:
        float: MSE value for the given shift.

    Example:
        >>> error = objective_function([1, -2], fcst, obs, ['x', 'y'])
    """
    # Validate input
    if len(shifts) != 2 or np.any(np.isnan(shifts)):
        return np.inf

    try:
        shift_x, shift_y = int(round(shifts[0])), int(round(shifts[1]))
    except (ValueError, TypeError):
        return np.inf

    shifted_fcst = shift_fcst(fcst, shift_x, shift_y, spatial_dims)

    # Mask forecast using fixed obs mask
    fcst_masked = shifted_fcst.where(fixed_mask)
    obs_masked = obs.where(fixed_mask)

    # To avoid artefacts, we allow shifts as long as the main forecast blob
    # (at least 80%) stays within the valid observation region
    valid_fraction = np.sum(~np.isnan(fcst_masked)) / np.sum(~np.isnan(fcst))

    if valid_fraction < 0.8:
        return np.inf

    mse_val = calc_mse(fcst_masked, obs_masked)
    corr_val = calc_corr_coeff(fcst_masked, obs_masked)

    # Penalize low correlation
    penalty = 1e3 if np.isnan(corr_val) or corr_val < 0.3 else 0

    return mse_val+penalty


def calc_mse_volume(shifted_fcst: xr.DataArray, obs: xr.DataArray) -> float:
    """
    Calculate the volume error component of CRA score.

    $$\text{Volume Error} = (\bar{f} - \bar{o})^2$$

    where $\bar{f}$ and $\bar{o}$ are the mean values of forecast and observation respectively.

    Args:
        shifted_fcst (xr.DataArray): Shifted forecast field.
        obs (xr.DataArray): Observation field.

    Returns:
        float: Volume error.

    Example:
        >>> volume_error = calc_mse_volume(shifted_fcst, obs)
    """
    mean_shifted_forecast = float(shifted_fcst.mean(skipna=True).values)
    mean_observed = float(obs.mean(skipna=True).values)
    volume_error = float((mean_shifted_forecast - mean_observed) ** 2)
    return volume_error


def calc_num_points(data: xr.DataArray, threshold: float) -> int:
    """
    Count the number of grid points in the data above a given threshold.

    Args:
        data (xr.DataArray): Input data array.
        threshold (float): Threshold value.

    Returns:
        int: Number of points above the threshold.

    Example:
        >>> count = calc_num_points(data, threshold=5.0)
    """
    mask = data >= threshold
    count_above_threshold = mask.sum().item()
    return count_above_threshold


def calc_corr_coeff(data1: xr.DataArray, data2: xr.DataArray) -> float:
    """
    Calculate the Pearson correlation coefficient between two data arrays.

    Args:
        data1 (xr.DataArray): First data array.
        data2 (xr.DataArray): Second data array.

    Returns:
        float: Correlation coefficient.

    Example:
        >>> corr = calc_corr_coeff(data1, data2)
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

def calc_resolution(obs: xr.DataArray, spatial_dims) -> float:
    y_coords = obs.coords[spatial_dims[0]].values
    x_coords = obs.coords[spatial_dims[1]].values

    # Compute resolution in native units
    dy = np.mean(np.diff(y_coords))
    dx = np.mean(np.diff(x_coords))

    # Check if coordinates are in degrees
    if np.all(np.abs(y_coords) <= 90) and np.all(np.abs(x_coords) <= 180):
        lat_mean = np.mean(y_coords)
        dy_km = np.abs(dy) * 111
        dx_km = np.abs(dx) * 111 * np.cos(np.radians(lat_mean))
    else:
        # Assume coordinates are in meters
        dy_km = np.abs(dy) / 1000
        dx_km = np.abs(dx) / 1000

    avg_resolution_km = np.mean([dy_km, dx_km])
    return float(avg_resolution_km)

def cra(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
    spatial_dims: Tuple[str, str],
    max_distance: float = 300
) -> Optional[dict]:
    """
    Compute the Contiguous Rain Area (CRA) score between forecast and observation fields.


    The CRA score decomposes the total mean squared error (MSE) into three components:
    displacement, volume, and pattern. It identifies contiguous rain blobs above a threshold,
    shifts the forecast blob to best match the observed blob, and evaluates the error reduction.


    The decomposition is defined as:


    .. math::
        \text{MSE}_{\text{total}} = \text{MSE}_{\text{displacement}} + \text{MSE}_{\text{pattern}} + \text{MSE}_{\text{volume}}

    where:
    - `MSE_displacement` is the error reduction due to optimal spatial alignment,
    - `MSE_volume` is the error due to differences in blob intensity,
    - `MSE_pattern` is the residual error after alignment and volume adjustment.

    This metric is spatial and blob-based, and does not support `reduce_dims`, `preserve_dims`, or `weights`.

    Args:
        fcst (xr.DataArray): Forecast field as an xarray DataArray.
        obs (xr.DataArray): Observation field as an xarray DataArray.
        threshold (float): Threshold to define contiguous rain areas.
        spatial_dims: A pair of dimension names ``(y, x)``
            E.g. ``('projection_y_coordinate', 'projection_x_coordinate')``, ``("lat","lon")``.

    Returns:
        dict: A dictionary containing the following CRA components and diagnostics:
            - mse_total (float): Total mean squared error between forecast and observed blobs.
            - mse_displacement (float): MSE due to spatial displacement between forecast and observed blobs.
            - mse_volume (float): MSE due to volume differences.
            - mse_pattern (float): MSE due to pattern/structure differences.
            - fcst_blob (xr.DataArray): Forecast data blob above the threshold.
            - obs_blob (xr.DataArray): Observed data blob above the threshold.
            - shifted_fcst (xr.DataArray): Forecast blob shifted to best align with observed blob.
            - optimal_shift (list[int]): Optimal [x, y] shift applied to forecast blob in grid points units.
            - num_gridpoints_above_threshold_fcst (int): Number of grid points in forecast blob above threshold.
            - num_gridpoints_above_threshold_obs (int): Number of grid points in observed blob above threshold.
            - avg_fcst (float): Mean value of the forecast blob.
            - avg_obs (float): Mean value of the observed blob.
            - max_fcst (float): Maximum value in the forecast blob.
            - max_obs (float): Maximum value in the observed blob.
            - corr_coeff_original (float): Correlation coefficient between original forecast and observed blobs.
            - corr_coeff_shifted (float): Correlation coefficient between shifted forecast and observed blobs.
            - rmse_original (float): Root mean square error between original forecast and observed blobs.
            - rmse_shifted (float): Root mean square error between shifted forecast and observed blobs.
        Returns None if input data is invalid or CRA computation fails.

    Raises:
        ValueError: If input shapes do not match or blobs cannot be computed.
        TypeError: If inputs are not xarray DataArrays.

    References:
        Ebert, E. E., & McBride, J. L. (2000). Verification of precipitation forecasts from operational numerical weather prediction models. *Weather and Forecasting*, 15(3), 247-263. https://doi.org/10.1016/S0022-1694(00)00343-7

    Example:
        >>> from scores.spatial import cra
        >>> import xarray as xr
        >>> fcst = xr.DataArray(...)  # your forecast data
        >>> obs = xr.DataArray(...)   # your observation data
        >>> result = cra(fcst, obs, threshold=5.0)
        >>> print(result['mse_total'])

    """

    # Type and shape checks
    if not isinstance(fcst, xr.DataArray):
        raise TypeError("fcst must be an xarray DataArray")
    if not isinstance(obs, xr.DataArray):
        raise TypeError("obs must be an xarray DataArray")
    if fcst.shape != obs.shape:
        raise ValueError("fcst and obs must have the same shape")

    for dim in spatial_dims:
        if dim not in obs.dims:
            raise ValueError(f"Spatial dimension '{dim}' not found in observation data")

    [fcst_blob, obs_blob] = generate_blobs(fcst, obs, threshold)

    if fcst_blob is None or obs_blob is None:
        return None

    mse_total = calc_mse(fcst_blob, obs_blob)

    [shifted_fcst, optimal_shift] = calc_shifted_forecast(fcst_blob, obs_blob, spatial_dims, max_distance)

    if shifted_fcst is None:
        return None

    mse_shift = calc_mse(shifted_fcst, obs_blob)
    mse_displacement = mse_total - mse_shift
    mse_volume = calc_mse_volume(shifted_fcst, obs_blob)
    mse_pattern = mse_shift - mse_volume

    if (
        mse_displacement < 0
        or np.isnan(mse_displacement)
        or mse_displacement == mse_total
        or mse_pattern < 0
        or abs(optimal_shift[0]) > 50
        or abs(optimal_shift[1]) > 50
    ):
        return None

    num_gridpoints_above_threshold_fcst = calc_num_points(fcst_blob, threshold)
    num_gridpoints_above_threshold_obs = calc_num_points(obs_blob, threshold)

    cra_dict = {
        "mse_total": mse_total,
        "mse_displacement": mse_displacement,
        "mse_volume": mse_volume,
        "mse_pattern": mse_pattern,
        "fcst_blob": fcst_blob,
        "obs_blob": obs_blob,
        "shifted_fcst": shifted_fcst,
        "optimal_shift": optimal_shift,
        "num_gridpoints_above_threshold_fcst": num_gridpoints_above_threshold_fcst,
        "num_gridpoints_above_threshold_obs": num_gridpoints_above_threshold_obs,
        "avg_fcst": float(np.mean(fcst_blob)),
        "avg_obs": float(np.mean(obs_blob)),
        "max_fcst": float(np.max(fcst_blob)),
        "max_obs": float(np.max(obs_blob)),
        "corr_coeff_original": calc_corr_coeff(fcst_blob, obs_blob),
        "corr_coeff_shifted": calc_corr_coeff(shifted_fcst, obs_blob),
        "rmse_original": calc_rmse(fcst_blob, obs_blob),
        "rmse_shifted": calc_rmse(shifted_fcst, obs_blob),
    }
    return cra_dict
