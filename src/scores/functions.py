"""
Contains functions which transform data or perform calculations
"""

from typing import overload

import numpy as np
import xarray as xr

from scores.typing import XarrayLike


def create_latitude_weights(latitudes):
    """
    A common way of weighting errors is to make them proportional to the amount of area
    which is contained in a particular region. This is approximated by the cosine
    of the latitude on an LLXY grid. Nuances not accounted for include the variation in
    latitude across the region, or the irregularity of the surface of the earth.

    Returns:
        An xarray containing the weight values to be used for area approximation

    Args:
        An xarray (or castable type) containing latitudes between +90 and -90 degrees

    Note - floating point behaviour can vary between systems, precisions and other factors
    """
    weights = np.cos(np.deg2rad(latitudes))
    return weights


# Dataset input types lead to a Dataset return type
@overload
def angular_difference(source_a: xr.Dataset, source_b: xr.Dataset) -> xr.Dataset: ...


# DataArray input types lead to a DataArray return type
@overload
def angular_difference(source_a: xr.DataArray, source_b: xr.DataArray) -> xr.DataArray:  # type: ignore
    ...


def angular_difference(source_a: XarrayLike, source_b: XarrayLike) -> XarrayLike:
    """
    Determines, in degrees, the smaller of the two explementary angles between
    two sources of directional data (e.g. wind direction).

    Args:
        source_a: direction data in degrees, first source
        source_b: direction data in degrees, second source

    Returns:
        An array containing angles within the range [0, 180].
    """
    difference = np.abs(source_a - source_b) % 360
    difference = difference.where(difference <= 180, 360 - difference)  # type: ignore
    return difference  # type: ignore
