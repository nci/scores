from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage
import xarray as xr
from scipy.optimize import minimize

from scores.continuous.standard_impl import mse, rmse


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
        print("Less than 10 points meet the condition.")
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
    fcst: xr.DataArray, obs: xr.DataArray, spatial_dims: List[str]
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

    # Compute original correlation coefficient
    valid_mask_orig = ~np.isnan(obs) & ~np.isnan(fcst)
    if not valid_mask_orig.any():
        print("No valid overlap in original data.")
        return None, None

    cc_orig = np.corrcoef(obs.values[valid_mask_orig], fcst.values[valid_mask_orig])[0, 1]

    if np.isnan(cc_orig):
        print("Original correlation coefficient is NaN.")
        return None, None

    fcst_bounding_box_centre = calc_bounding_box_centre(fcst)  # [y,x]
    obs_bounding_box_centre = calc_bounding_box_centre(obs)

    initial_shift = [
        obs_bounding_box_centre[1] - fcst_bounding_box_centre[1],  # x_shift
        obs_bounding_box_centre[0] - fcst_bounding_box_centre[0],  # y_shift
    ]
    result = minimize(objective_function, initial_shift, args=(fcst, obs, spatial_dims), method="Nelder-Mead")
    optimal_shift = result.x
    if np.any(np.isnan(optimal_shift)) or result.fun == np.inf:
        # print("Optimization failed: invalid shift or no valid overlap")
        return None, None

    # Apply the optimal shift

    shift_kwargs = {
        spatial_dims[0]: int(optimal_shift[1]),  # spatial_dims[0] -> y
        spatial_dims[1]: int(optimal_shift[0]),  # spatial_dims[1] -> x
    }

    shifted_fcst_values = fcst.shift(**shift_kwargs, fill_value=np.nan)

    shifted_fcst = xr.DataArray(shifted_fcst_values, dims=fcst.dims, coords=fcst.coords)

    return shifted_fcst, [int(x) for x in optimal_shift]


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
        >>> shifted = shift_fcst(fcst, 2, -1, ['x', 'y'])
    """
    shift_kwargs = {
        spatial_dims[0]: int(shift_x),
        spatial_dims[1]: int(shift_y),
    }
    return fcst.shift(
        **shift_kwargs,
        fill_value=np.nan,
    )


def objective_function(shifts: List[int], fcst: xr.DataArray, obs: xr.DataArray, spatial_dims: List[str]) -> float:
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

    shift_x, shift_y = shifts
    shifted_fcst = shift_fcst(fcst, shift_x, shift_y, spatial_dims)

    # Ensure valid comparison
    valid_mask = ~np.isnan(shifted_fcst) & ~np.isnan(obs)
    if not valid_mask.any():
        return np.inf  # No overlap, return large error
    return calc_mse(shifted_fcst.values[valid_mask], obs.values[valid_mask])


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


def _infer_spatial_dims(dataarray: xr.DataArray) -> List[str]:
    """
    Infer spatial dimensions from an xarray DataArray based on common naming conventions.

    Args:
        dataarray (xr.DataArray): Input data array.

    Returns:
        list[str]: List of inferred spatial dimension names.

    Example:
        >>> dims = _infer_spatial_dims(data)
    """

    spatial_keywords = {
        "x",
        "y",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "projection_x_coordinate",
        "projection_y_coordinate",
        "grid_latitude",
        "grid_longitude",
    }
    return [dim for dim in dataarray.dims if dim.lower() in spatial_keywords]


def cra(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
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

    # Infer spatial dimensions
    spatial_dims = _infer_spatial_dims(fcst)
    if len(spatial_dims) != 2:
        raise ValueError("Could not infer exactly two spatial dimensions from input data")

    for dim in spatial_dims:
        if dim not in obs.dims:
            raise ValueError(f"Spatial dimension '{dim}' not found in observation data")

    [fcst_blob, obs_blob] = generate_blobs(fcst, obs, threshold)

    if fcst_blob is None or obs_blob is None:
        return None

    mse_total = calc_mse(fcst_blob, obs_blob)

    [shifted_fcst, optimal_shift] = calc_shifted_forecast(fcst_blob, obs_blob, spatial_dims)

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
