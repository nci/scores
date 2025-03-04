import pytest
import numpy as np
import xarray as xr

from scores.continuous import nse
from scores.utils import DimensionError


class NseSetup:
    """
    Base class for NSE tests with some setup and helper functions
    """
    _SEED: int = 42

    @staticmethod
    def make_random_xr_array(*, shape: tuple[int], dim_names: list[str]):
        """
        Random xarray data array with each element in multi-index, "i", normally distributed,
        math:`X_i ~ N(0, 1)`pi.  ``dim_names`` must match the size of ``shape``.
        """
        return xr.DataArray(np.random.rand(*shape), dims=dim_names)

    @pytest.fixture(scope="class", autouse=True)
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
    def test_error_incompatible_fcst_obs_dimnames(self):
        fcst = NseSetup.make_random_xr_array(
            shape=(4,2,3),
			dim_names=["t1", "x1", "y1"],
         )

        obs = NseSetup.make_random_xr_array(
            shape=(4,2,3),
			dim_names=["t1", "x2", "y2"],
         )

        with pytest.raises(DimensionError):
            nse(fcst, obs)

    def test_error_incompatible_weight_dims(self):
        reduce_dims = "t"

        fcst = NseSetup.make_random_xr_array(
            shape=(4,2,3),
			dim_names=["t", "x", "y"],
         )

        obs = NseSetup.make_random_xr_array(
            shape=(4,2,3),
			dim_names=["t", "x", "y"],
         )

        weights = NseSetup.make_random_xr_array(
            shape=(5,1,2),
            dim_names=["t", "x", "y"],
         )

        with pytest.raises(ValueError):
            res = nse(fcst, obs, weights=weights, reduce_dims=reduce_dims)

    def test_error_invalid_reduce_dims(self):

        fcst = NseSetup.make_random_xr_array(
            shape=(4,2,3),
			dim_names=["t", "x", "y"],
         )

        obs = NseSetup.make_random_xr_array(
            shape=(4,2,3),
			dim_names=["t", "x", "y"],
         )

        # does not exist
        reduce_dims = [ "potato" ]

        with pytest.raises(ValueError):
            res = nse(fcst, obs, reduce_dims=reduce_dims)

        # overspecified
        reduce_dims = [ "t", "x", "y", "w" ]

        with pytest.raises(ValueError):
            res = nse(fcst, obs, reduce_dims=reduce_dims)

    def test_error_no_dim_reduced(self):
        pass

    def test_error_reduced_dims_not_enough_data_for_obs_variance(self):
        pass

    def test_warn_negative_weights(self):
        pass

    def test_warn_divide_by_zero(self):
        pass

    def test_success_scalar(self):
        pass

    def test_success_1d_array(self):
        pass

    def test_success_nd_array(self):
        pass

    def test_success_nd_array_weights(self):
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

    @staticmethod
    def naive_nse(fcst: np.float64, obs: np.float64, weight: np.float64):
        pass


class TestNseDask(NseSetup):
    """
    Basic testing if dask is available and used appropriately by NSE.
    """
    @pytest.fixture(scope="class", autouse=True)
    def dask_available(self):
        try:
            import dask
            import dask.array
        except ImportError:
            pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

