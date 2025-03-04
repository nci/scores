import pytest
import numpy as np
from numpy import typing as npt
import xarray as xr

from scores.continuous import nse
from scores.utils import DimensionError


class NseSetup:
    """
    Base class for NSE tests with some setup and helper functions
    """
    _SEED: int = 42

    @staticmethod
    def make_random_xr_array(shape: tuple[int], dim_names: list[str]) -> xr.DataArray:
        """
        Random xarray data array with each element in multi-index, "i", normally distributed,
        math:`X_i ~ N(0, 1)`pi.  ``dim_names`` must match the size of ``shape``.
        """
        return xr.DataArray(np.random.rand(*shape), dims=dim_names)

    @staticmethod
    def make_xr_array_all_ones(shape: tuple[int], dim_names: list[str]) -> xr.DataArray:
        """
        Array with all ones, used to mimic divide by zero conditions where the observation variance
        and/or forecast error are zero.
        """
        return xr.DataArray(np.ones(shape), dims=dim_names)

    @staticmethod
    def naive_nse(
        *,
        fcst: npt.NDArray[float],
        obs: npt.NDArray[float],
        weight: npt.NDArray[float],
    ):
        """
        Naive implementation of NSE using for loops - this is to check that the internals of e.g.
        xarray/numpy/dask are doing the right thing in conjunction with how they are used for this
        score. However, this function is slow and should not be run for big arrays.
        """
        pass

    @pytest.fixture
    def setup_numpy_seed(self):
        """
        Auto-reset numpy seed for each test that inherits this base class
        """
        np.random.seed(NseSetup._SEED)


class TestNsePublicApi(NseSetup):
    """
    Test suite that tests the public API. Mainly consists of structural tests and argument
    compatibility, as well as checking expected errors and warnings are raised. For specific scoring
    tests ``TestNseScore`` is more suited.


    NOTE: this implicitly tests most of the ``NseScoreBuilder`` class, since it essentially
    dispatches any checks to that class, as such they are not repeated as long as coverage is happy.
    """

    @classmethod
    def setup_class(cls):
        """
        Common data to be reused by the rest of this class
        """
        # --- input arrays ---
        # default obs
        cls.obs = NseSetup.make_random_xr_array(
            shape=(4,2,3),
            dim_names=["t", "x", "y"],
        )
        # default fcst 
        cls.fcst = NseSetup.make_random_xr_array(
            shape=(4,2,3),
            dim_names=["t", "x", "y"],
        )
        # bad obs name
        # note: all the names must be wrong, if some names are correct
        cls.obs_badnames = NseSetup.make_random_xr_array(
            shape=(4,2,3),
            dim_names=["t_bad", "x_bad", "y_bad"],
        )
        # incorrect dim sizes between arrays (non broadcastable)
        cls.obs_baddimsizes = NseSetup.make_random_xr_array(
            shape=(5,1,2),
            dim_names=["t", "x", "y"],
        )

        # --- dimension specification ---
        # good dimension specifications
        cls.reduce_dim_string = "t"           # should give (x, y) array
        cls.reduce_dims_list = [ "t", "x" ]   # should give (y, 1) array
        cls.reduce_dim_all = "all"            # single value output
        cls.preserve_dim_string = "y"         # should give (y, 1) array
        cls.preserve_dim_list = [ "x", "y" ]  # should give (x, y) array
        # bad dimension specifications
        cls.preserve_dim_all1 = "all"          # NSE cannot be computed, no dimensions reduced 
        cls.preserve_dim_all2 = [ "x", "y", "t" ]  # ditto above
        cls.reduce_dims_none = [ ]             # ditto above

        # --- insufficient data to reduce ---
        cls.fcst_insufficient_data  = NseSetup.make_random_xr_array(
            shape=(4,1,1),
            dim_names=["t", "x", "y"],
        )
        cls.obs_insufficient_data = NseSetup.make_random_xr_array(
            shape=(4,1,1),
            dim_names=["t", "x", "y"],
        )

        # --- weights ---
        # default weights
        #   NOTE: ideally weights should be specified explicitly for each index, but broadcasting is
        #   currently allowed (and potentially okay, as long as the dimension shapes conform for
        #   the specified axes.)
        cls.weights = NseSetup.make_random_xr_array(
            shape=(4,2),
            dim_names=["t", "x"],
        )
        # these should still work, as "nan" acts as exclusion and 0 acts as zero-forcing
        cls.weights.loc[1, "x"] = 0.0     # make one of the weights 0 
        cls.weights.loc[1, "t"] = np.nan  # make one of the weights np.nan 
        # failure conditions for weights
        cls.negative_weights = cls.weights.copy(deep=True) 
        cls.negative_weights.loc[0, "t"] = -1.0  # make one of the weights negative
        cls.allzero_weights = cls.weights.copy(deep=True, data=0.0) 
        cls.allnan_weights = cls.weights.copy(deep=True, data=np.nan)

        # --- divide by zero ---
        # set the value along one index of the dimension being reduced: "t", as all one
        # this will force a divide by zero error in one entry in the resulting array, which should
        # still allow the calculation to be completed.

        # obs_divide_by_zero v.s. fcst (default) = -np.inf only on the plane where this happens
        cls.obs_divide_by_zero = cls.obs.copy(deep=True)
        cls.obs_divide_by_zero.loc[dict(t=1)] = 42.123 
        # obs_divide_by_zero v.s. fcst_divide_by_zero = np.nan, ditto above
        cls.fcst_divide_by_zero = cls.fcst.copy(deep=True)
        cls.fcst_divide_by_zero.loc[dict(t=1)] = 42.123

    def test_error_incompatible_fcst_obs_dimnames(self):
        with pytest.raises(DimensionError):
            nse(self.fcst, self.obs_badnames)

    def test_error_incompatible_dimsizes(self):
        with pytest.raises(ValueError):
            nse(fcst, self.obs_baddimsizes, reduce_dims=self.reduce_dim_string)

    def test_error_invalid_dims_specification(self):
        # preserve=all
        with pytest.raises(DimensionError):
            nse(fcst, self.obs, preserve_dims=self.preserve_dim_all1)
        # essentially the same as above but explicitly specified - see class setup
        with pytest.raises(DimensionError):
            nse(fcst, self.obs, preserve_dims=self.preserve_dim_all2)
        # reduce_dims is empty - also essentially the same
        with pytest.raises(DimensionError):
            nse(fcst, self.obs, reduce_dims=self.reduce_dims_none)
        # both options specified (safety check)
        with pytest.raises(DimensionError):
            nse(fcst, self.obs, reduce_dims=self.reduce_dim_string, preserve_dims=self.preserve_dim_list)

    def test_error_no_dim_reduced(self):
        pass

    def test_error_reduced_dims_not_enough_data_for_obs_variance(self):
        pass

    def test_warn_negative_weights(self):
        pass

    def test_warn_divide_by_zero(self):
        pass



class TestNseScoreBuilder:
    """
    NOTE: most of NseScoreBuilder is tested by the public API test suite and is not repeated here.
    Only things missed by the public API test suite will be covered here. 
    """
    ...


class TestNseUtil(NseSetup):
    """
    NOTE: most of NseScoreBuilder is tested by the public API test suite and is not repeated here.
    Only things missed by the public API test suite will be covered here. 
    """
    ...


class TestNseScore(NseSetup):
    """
    Tests validity of the actual NSE scoring functions for a given set of inputs. As such, tests
    here should not use ``np.random.random``, instead they'd need to be either handcrafted or
    verifiable by a naive secondary algorithm.
    """

class TestNseDataset(NseSetup):
    """
    Basic testing for compatibility with xarray datasets. Only variables & dimensions that match
    between datasets will be computed. This is just a safety test to see that NSE still works fine
    with datasets.

    NOTE: failure conditions will not be the responsibility of this test, as there are utility
    functions that should handle this.
    """
    def test_nse_with_datasets(self):
        """
        obs:
        temp = x, y, t
        precip = x, y, t
        ---
        fcst:
        temp = x, y, t, h
        precip = t, h
        tapioca = x, y
        ---
        reduce_dims = [ "x", "y" ]
        ---
        expected behaviour:
        - obs & forecast should be broadcast appropriately so that they both contain x,y,t,h for both arrays.
        - reduce_dims then applies as per normal
        - tapioca is ignored
        - no errors
        """
        ds_obs = xr.Dataset(
            data_vars=dict(
                temp=NseSetup.make_random_xr_array((3,5,2), ["x", "y", "t"]),
                precip=NseSetup.make_random_xr_array((3,5,2), ["x", "y", "t"]),
            ),
        )
        ds_fcst = xr.Dataset(
            data_vars=dict(
                temp=NseSetup.make_random_xr_array((3,5,2,4), ["x", "y", "t", "h"]),
                precip=NseSetup.make_random_xr_array((3,5), ["x", "y"]),
                tapioca=NseSetup.make_random_xr_array((2,5), ["t", "y"]),
            ),
        )
        reduce_dims = [ "x", "y" ]
        res = nse(ds_fcst, ds_obs, reduce_dims=reduce_dims)


class TestNseDask(NseSetup):
    """
    Basic testing if dask is available and used appropriately by NSE.

    NOTE: failure conditions will not be the responsibility of this test, this suite just exists to
    check if dask computes things appropriately with non-dask as a compatiblity measure.
    """
    @pytest.fixture(scope="class", autouse=True)
    def skip_if_dask_unavailable(self):
        try:
            import dask
            import dask.array
        except ImportError:
            pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

