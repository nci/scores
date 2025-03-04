"""
Test suite that tests helper functions and classes in the ``scores.typing`` module.

In particular:
    - tests functionality for consolidating ``xr.Datasets`` and ``xr.DataArray`` so they can
      operate as a single type.
    - tests type homogeneity between different ``XarrayLike`` types after conversion to
      ``LiftedDataset``(s)
    - checks that any conversion is isomorphic (i.e. structure preserving).
"""

import pytest
import xarray as xr

import scores.typing

_DS_TEST = xr.Dataset({"a": xr.DataArray([1], dims="t"), "b": xr.DataArray([2], dims="t")})


def test_lift_dataarray_withname():
    """
    Checks isomorphism for data arrays with names

    Also checks that the LiftedDataset has correctly populated metadata.
    """
    # technically it doesn't matter if the DataArray name is the same as the arbitrary dummy name,
    # since LiftedDataset.dummy (bool) is what matters, but still would be good to not have the
    # same array name for the test.
    da_test_name = "potato"
    assert da_test_name != scores.typing.LiftedDataset._DUMMY_DATAARRAY_VARNAME  # pylint: disable-msg=protected-access
    da = xr.DataArray([1], dims="t", name=da_test_name)
    lds = scores.typing.LiftedDataset(da)
    # check lifted dataset only has one array
    keys = list(lds.ds.variables.keys())
    assert len(keys) == 1
    # check that the name is preserved and isn't a dummy
    assert keys[0] == da_test_name
    assert not lds.dummy_name
    # check that marker is XarrayTypeMarker.DATAARRAY
    assert lds.xr_type_marker == scores.typing.XarrayTypeMarker.DATAARRAY
    # check raw format is preserved when converted back to dataarray with original name
    da_raw = lds.raw()
    assert da_raw.identical(da)


def test_lift_dataarray_withnameequalsdummy():
    """
    Checks isomorphism for data arrays with names, even if for some odd reason they have the same
    dummy name.

    Also checks that the LiftedDataset has correctly populated metadata.
    """
    # now intentionally set the test name to the dummy name
    da_test_name = scores.typing.LiftedDataset._DUMMY_DATAARRAY_VARNAME  # pylint: disable-msg=protected-access
    da = xr.DataArray([1], dims="t", name=da_test_name)
    lds = scores.typing.LiftedDataset(da)
    # check lifted dataset only has one array
    keys = list(lds.ds.variables.keys())
    assert len(keys) == 1
    # check that the name is the same as the dummy, BUT it isn't actually flagged as a dummy
    assert keys[0] == scores.typing.LiftedDataset._DUMMY_DATAARRAY_VARNAME  # pylint: disable-msg=protected-access
    assert not lds.dummy_name
    # check that marker is XarrayTypeMarker.DATAARRAY
    assert lds.xr_type_marker == scores.typing.XarrayTypeMarker.DATAARRAY
    # check raw format is preserved when converted back to dataarray with original name
    da_raw = lds.raw()
    assert da_raw.identical(da)


def test_lift_dataarray_noname():
    """
    Checks isomorphism for data arrays with no names
    """
    # now we have no name
    da = xr.DataArray([1], dims="t")
    lds = scores.typing.LiftedDataset(da)
    # check lifted dataset only has one array
    keys = list(lds.ds.variables.keys())
    assert len(keys) == 1
    # check that the name is the same as the dummy, AND is also flagged as dummy
    assert keys[0] == scores.typing.LiftedDataset._DUMMY_DATAARRAY_VARNAME  # pylint: disable-msg=protected-access
    assert lds.dummy_name
    # check that marker is XarrayTypeMarker.DATAARRAY
    assert lds.xr_type_marker == scores.typing.XarrayTypeMarker.DATAARRAY
    # check raw format is preserved when converted back to dataarray with no name
    da_raw = lds.raw()
    assert da_raw.identical(da)


def test_lift_dataset():
    """
    Checks isomorphism for datasets.

    Also checks that the LiftedDataset has correctly populated metadata.
    """
    ds = _DS_TEST
    # check lifted dataset has all its variables
    lds = scores.typing.LiftedDataset(ds)
    # check that marker is XarrayTypeMarker.DATASET
    assert lds.xr_type_marker == scores.typing.XarrayTypeMarker.DATASET
    # check that name isn't dummy
    assert not lds.dummy_name
    # check raw format is preserved when converted back to dataarray with no name
    ds_raw = lds.raw()
    assert ds_raw.identical(ds)


def test_invalid_input_type():
    """
    Checks that invalid types raise an error, since this is not a anemic data model.
    """
    # lists are not lifted automatically, only `xr.DataArrays`
    with pytest.raises(TypeError):
        scores.typing.LiftedDataset([1, 2, 3])


@pytest.mark.parametrize(
    "xr_data, do_lift, expect_error_raised",
    [
        # one dataset and one data array => should produce error
        (tuple([xr.DataArray([1], dims="a"), _DS_TEST]), True, True),
        # multiple inputs with data arrays => all data arrays => no error
        (tuple([xr.DataArray([1], dims="a"), xr.DataArray([2], dims="b")]), True, False),
        # single input with dataset => all datasets => no error
        (tuple([_DS_TEST]), True, False),
        # intentionally don't lift the dataset to prompt a type error => should produce error
        (tuple([_DS_TEST]), False, True),
    ],
)
def test_same_lds_type(xr_data, do_lift, expect_error_raised):
    """
    Checks for homogeneity in LiftedDataset (lds) types, and checks that it raises an error if they
    are of different types.
    """
    lds = [scores.typing.LiftedDataset(x) for x in xr_data] if do_lift else xr_data
    if expect_error_raised:
        with pytest.raises(TypeError):
            scores.typing.check_lds_same_type(*lds)
    else:
        scores.typing.check_lds_same_type(*lds)
