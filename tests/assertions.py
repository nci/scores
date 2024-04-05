"""
This module containts various complex assertion routines which are used in unit tests
"""

import numpy as np
import xarray as xr


# pylint: disable=too-many-arguments
def assert_dataarray_equal(
    left,
    right,
    check_name=True,
    check_attrs_values=True,
    check_attrs_order=False,
    check_dtype=True,
    decimals=None,
):
    """
    Check that two Xarray DataArrays are equal.

    Args:
        left (xarray.DataArray): a DataArray
        right (xarray.DataArray): another DataArray
        check_name (Optional[bool]): whether to check the name property of
            the DataArrays, defaults to True
        check_attrs_values (Optional[bool]): whether to check DataArray
            attributes, defaults to True
        check_attrs_order (Optional[bool]): whether to check the order of the
            DataArray attributes. The order can be checked without checking
            the values. Defaults to False
        check_dtype(Optional[bool]): If True (default), then checks whether
            DataArrays have the same dtype.
        decimals (Optional[int]): If supplied, then the DataArrays are rounded
            to this many decimal places when testing equality (see np.round).

    Returns:
        None
    """
    if not isinstance(left, xr.DataArray):
        raise TypeError(f"left must be an xarray.DataArray, not {type(left)}")
    if not isinstance(right, xr.DataArray):
        raise TypeError(f"right must be an xarray.DataArray, not {type(right)}")

    # check that the types of left and right are compatible
    assert isinstance(left, type(right)) or isinstance(
        right, type(left)
    ), f"Incompatible types: left: {type(left)}, right: {type(right)}"

    # remember the object type
    prefix = type(left).__name__

    # if decimals are supplied, do a rounding, otherwise rounding is just a dummy 'identity' func
    # pylint: disable=unnecessary-lambda-assignment
    rounding = lambda x: x if decimals is None else np.round(x, decimals=decimals)  # noqa: E731

    decimal_str = "" if decimals is None else f" to {decimals} decimal places"

    # check the values using xarray.DataArray.equals or xarray.Dataset.equals
    assert rounding(left).equals(
        rounding(right)
    ), f"{prefix}s are not equal{decimal_str}: \nleft: {left}\nright: {right}\n"

    # check the Dataset or DataArray attributes
    if check_attrs_values:
        np.testing.assert_equal(
            left.attrs,
            right.attrs,
            err_msg=f"{prefix} attributes are not equal",
        )

    # check the attributes order
    if check_attrs_order:
        left_keys = list(left.attrs.keys())
        right_keys = list(right.attrs.keys())
        assert left_keys == right_keys, (
            f"order of {prefix} attributes are different:\n" f"left: {left_keys}\n" f"right: {right_keys}\n"
        )

    # check that the names of the dataarrays are equal
    if check_name:
        assert left.name == right.name, f"DataArray names are not equal:\nleft: {left.name}\nright: {right.name}\n"

    if check_dtype:
        assert left.dtype == right.dtype, (
            f"DataArray dtypes are not equal:\nleft: {left.dtype}\nright: " f"{right.dtype}\n"
        )


# pylint: disable=too-many-arguments
def assert_dataset_equal(
    left,
    right,
    check_ds_attrs_values=True,
    check_da_attrs_values=True,
    check_attrs_order=False,
    check_dtype=True,
    decimals=None,
):
    """
    Assert that two Xarray datasets are equal

    Args:
        left (xarray.Dataset): a Dataset
        right (xarray.Dataset): another Dataset
        check_ds_attrs_values (Optional[bool]): whether to check the Dataset
            attributes, defaults to True
        check_da_attrs_values (Optional[bool]): whether to check the DataArray
            attributes of each data variable in the Datasets, defaults to True
        check_attrs_order (Optional[bool]): whether to check the order of the
            Dataset and/or DataArray attributes. The order can be checked
            without checking the values. Defaults to False
        check_dtype(Optional[bool]): If True (default), then checks whether
            the data variables have the same dtype.
        decimals (Optional[int]): If supplied, then the data variables are
            rounded to this many decimal places when testing equality
            (see np.round).

    Returns:
        None
    """
    if not isinstance(left, xr.Dataset):
        raise TypeError(f"left must be an xarray.Dataset, not {type(left)}")
    if not isinstance(right, xr.Dataset):
        raise TypeError(f"right must be an xarray.Dataset, not {type(right)}")

    _assert_xarray_equal(
        left,
        right,
        check_attrs_values=check_ds_attrs_values,
        check_attrs_order=check_attrs_order,
        decimals=decimals,
    )

    if check_da_attrs_values or check_attrs_order or check_dtype:
        for da_name in left.data_vars:
            da_left = left[da_name]
            da_right = right[da_name]
            try:
                assert_dataarray_equal(
                    da_left,
                    da_right,
                    check_attrs_values=check_da_attrs_values,
                    check_attrs_order=check_attrs_order,
                    check_dtype=check_dtype,
                    decimals=decimals,
                )
            except AssertionError as exc:
                raise AssertionError(f'Dataset variables "{da_name}" are not equal:\n{str(exc)}') from exc


# pylint: disable=unnecessary-lambda-assignment
def _assert_xarray_equal(left, right, check_attrs_values=True, check_attrs_order=False, decimals=None):
    """
    Check that two Xarray objects (Dataset or DataArray) are equal.

    Args:
        left (xarray.DataArray or xarray.Dataset): left object
        right (xarray.DataArray or xarray.Dataset): right object
        check_attrs_values (Optional[bool]): whether to check `OrderedDict`s
            `left.attrs` against `right.attrs`. Defaults to True
        check_attrs_order (Optional[bool]): whether to check the order of keys
            in the `OrderedDict`s `left.attrs` and `right.attrs`. The order
            can be checked without checking the values. Defaults to False
        decimals (Optional[int]): If supplied, then the data objects are
            rounded to this many decimal places (see np.round).

    Returns:
        None
    """

    # check that the types of left and right are compatible
    assert isinstance(left, type(right)) or isinstance(
        right, type(left)
    ), f"Incompatible types: left: {type(left)}, right: {type(right)}"

    # remember the object type
    prefix = type(left).__name__

    # if decimals are supplied, do a rounding, otherwise rounding is just a dummy 'identity' func
    rounding = lambda x: x if decimals is None else np.round(x, decimals=decimals)  # noqa: E731

    decimal_str = "" if decimals is None else f" to {decimals} decimal places"

    # check the values using xarray.DataArray.equals or xarray.Dataset.equals
    assert rounding(left).equals(
        rounding(right)
    ), f"{prefix}s are not equal{decimal_str}: \nleft: {left}\nright: {right}\n"

    # check the Dataset or DataArray attributes
    if check_attrs_values:
        np.testing.assert_equal(
            left.attrs,
            right.attrs,
            err_msg=f"{prefix} attributes are not equal",
        )

    # check the attributes order
    if check_attrs_order:
        left_keys = list(left.attrs.keys())
        right_keys = list(right.attrs.keys())
        assert left_keys == right_keys, (
            f"order of {prefix} attributes are different:\n" f"left: {left_keys}\n" f"right: {right_keys}\n"
        )
