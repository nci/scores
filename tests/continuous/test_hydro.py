import numpy as np
import pytest
import xarray as xr
from numpy import typing as npt

from scores.continuous import kge, nse, pbias
from scores.utils import DimensionError

from .test_standard import BIAS_WEIGHTS, DA1_BIAS, DA2_BIAS, DA3_BIAS, DS_BIAS1, DS_BIAS2

DASK_AVAILABLE = False
try:
    import dask
    import dask.array

    DASK_AVAILABLE = True
except ImportError:
    pass


# Metafunction used to generate tests from TestClasses
def pytest_generate_tests(metafunc):
    """
    Metafunction that looks through the reserved "params"  arg list of each test class

    Usage ::

        class Test...():

            params = {
                "test_1": dict(x=1, y=2),
                ...
            }

            def test_1(self, x, y):
                assert y != x

    Taken directly (and adapted slightly) from:
         doc: https://docs.pytest.org/en/stable/example/parametrize.html
         section: parametrizing-test-methods-through-per-class-configuration
    """
    # called once per each test function
    if hasattr(metafunc.cls, "params"):
        funcarglist = metafunc.cls.params[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [[funcargs[name] for name in argnames] for funcargs in funcarglist],
        )


class NseSetup:
    """
    Base class for NSE tests with some setup and helper functions
    """

    _SEED: int = 42

    @staticmethod
    def make_random_xr_array(
        shape: tuple[int, ...],
        dim_names: list[str],
        override_seed: int | None = None,
    ) -> xr.DataArray:
        """
        Random xarray data array with each element in multi-index, "i", normally distributed,
        math:`X_i ~ N(0, 1)`pi.  ``dim_names`` must match the size of ``shape``.

        Optional ``override_seed`` to change seed - caution this may be okay during setup e.g.
        in ``setup_class`` - especially if the forecast and obs need to use different seeds.

        .. caution::

            Do not use ``override_seed`` when generating random arrays INSIDE a test. During setup
            is okay...
        """
        if override_seed is not None:
            assert isinstance(override_seed, int)
            np.random.seed(override_seed)
        return xr.DataArray(np.random.rand(*shape), dims=dim_names)

    @staticmethod
    def make_xr_array_all_ones(shape: tuple[int], dim_names: list[str]) -> xr.DataArray:
        """
        Array with all ones, used to mimic divide by zero conditions where the observation variance
        and/or forecast error are zero.
        """
        return xr.DataArray(np.ones(shape), dims=dim_names)

    @staticmethod
    def nse_naive(
        fcst: npt.NDArray[float],
        obs: npt.NDArray[float],
        weights: npt.NDArray[float],
    ):
        """
        Naive implementation of NSE using for loops - this is to check that the internals of e.g.
        xarray/numpy/dask are doing the right thing in conjunction with how they are used for this
        score. However, this function is slow and should not be run for big arrays.

        used mainly by NseScore as a helper
        """
        assert fcst.shape == obs.shape
        assert weights.shape == fcst.shape
        ret_shape = (2, 4)
        obs_mean = np.zeros(ret_shape)
        fcst_error = np.zeros(ret_shape)
        obs_variance = np.zeros(ret_shape)
        # multindex : (0, 0, 0, 0) -> (2, 4, 2*, 3*): (*) => dim to be reduced
        #           : total iterations: 2 * 4 = 8
        #           : total broadcast elements per iteration = 2 * 3 = 6

        for idx in np.ndindex(ret_shape):
            obs_mean[idx] = np.mean(obs[idx])

        for idx in np.ndindex(fcst.shape):
            ix, iy, _, _ = idx
            _f, _o, _w = (fcst[idx], obs[idx], weights[idx])
            _om = obs_mean[ix, iy]
            fcst_error[ix, iy] += _w * np.power((_f - _o), 2)
            obs_variance[ix, iy] += _w * np.power(_om - _o, 2)

        nse_score = 1.0 - fcst_error / obs_variance

        return nse_score

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
    """

    _OBS_DEFAULT = NseSetup.make_random_xr_array(
        shape=(4, 2, 3),
        dim_names=["t", "x", "y"],
        override_seed=42,
    )
    _FCST_DEFAULT = NseSetup.make_random_xr_array(
        shape=(4, 2, 3),
        dim_names=["t", "x", "y"],
        override_seed=42,
    )
    _OBS_WRONG_DIMNAMES = NseSetup.make_random_xr_array(
        shape=(4, 2, 3),
        dim_names=["t_bad", "x_bad", "y_bad"],
    )
    _OBS_WRONG_DIMSIZES = NseSetup.make_random_xr_array(
        shape=(5, 1, 2),
        dim_names=["t", "x", "y"],
    )
    _OBS_INSUFFICIENT_DATA = NseSetup.make_random_xr_array(
        shape=(4, 1, 1),
        dim_names=["t", "x", "y"],
        override_seed=42,
    )
    _FCST_INSUFFICIENT_DATA = NseSetup.make_random_xr_array(
        shape=(4, 1, 1),
        dim_names=["t", "x", "y"],
        override_seed=24,
    )
    _WEIGHTS_DEFAULT = NseSetup.make_random_xr_array(
        shape=(4, 2),
        dim_names=["t", "x"],
    )
    _WEIGHTS_DEFAULT.loc[dict(x=1, t=0)] = 0.0
    _WEIGHTS_DEFAULT.loc[dict(t=1, x=0)] = np.nan
    _WEIGHTS_NEGATIVE = _WEIGHTS_DEFAULT.copy(deep=True)
    _WEIGHTS_NEGATIVE.loc[dict(t=0, x=0)] = -1.0
    _WEIGHTS_ALLZEROS = _WEIGHTS_DEFAULT.copy(
        deep=True,
        data=np.zeros(_WEIGHTS_DEFAULT.shape),
    )
    _WEIGHTS_ALLNANS = _WEIGHTS_DEFAULT.copy(
        deep=True,
        data=np.full(_WEIGHTS_DEFAULT.shape, fill_value=np.nan),
    )
    _OBS_DIVIDE_BY_ZERO = _OBS_DEFAULT.copy(deep=True)
    _OBS_DIVIDE_BY_ZERO.loc[dict(t=1)] = 42.123
    _FCST_DIVIDE_BY_ZERO = _FCST_DEFAULT.copy(deep=True)
    _FCST_DIVIDE_BY_ZERO.loc[dict(t=1)] = 42.123

    # reserved pytest name to dispatch params to tests
    params = {
        "test_error_incompatible_dims": [
            # incompatible dimension names
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_WRONG_DIMNAMES,
                reduce_dims="t",  # need to set this, otherwise "t_bad" maybe auto included
                preserve_dims=None,
                expect_context=pytest.raises(DimensionError),
            ),
            # incompatible dimension sizes
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_WRONG_DIMSIZES,
                reduce_dims="t",
                preserve_dims=None,
                expect_context=pytest.raises(ValueError),
            ),
            # preserve all
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                reduce_dims=None,
                preserve_dims="all",
                expect_context=pytest.raises(ValueError),
            ),
            # preserve all (explicitly specified)
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                reduce_dims=None,
                preserve_dims=["x", "y", "t"],
                expect_context=pytest.raises(ValueError),
            ),
            # no dims reduced - essentially the same as preserve all
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                reduce_dims=[],
                preserve_dims=None,
                expect_context=pytest.raises(ValueError),
            ),
            # overspecified - in theory this is valid, but in practice scores does
            # not attempt to resolve both reduce_dims AND preserve_dims - mutually
            # exclusive.
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                reduce_dims="t",
                preserve_dims=["x", "y"],
                expect_context=pytest.raises(ValueError),
            ),
        ],
        "test_error_insufficient_data": [
            dict(
                fcst=_FCST_INSUFFICIENT_DATA,
                obs=_OBS_INSUFFICIENT_DATA,
                reduce_dims=["x", "y"],
            ),
        ],
        "test_error_invalid_weights": [
            # any negative
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                weights=_WEIGHTS_NEGATIVE,
                reduce_dims=["x", "t"],
            ),
            # all zeros
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                weights=_WEIGHTS_ALLZEROS,
                reduce_dims=["x", "t"],
            ),
            # all nans
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                weights=_WEIGHTS_ALLNANS,
                reduce_dims=["x", "t"],
            ),
        ],
        "test_warn_divide_by_zero": [
            # 0 / 0 => should fill with nan
            dict(
                fcst=_FCST_DIVIDE_BY_ZERO,
                obs=_OBS_DIVIDE_BY_ZERO,
                reduce_dims=["x", "y"],
                both_zero=True,
            ),
            # a / 0 where a > 0 => should fill with -inf
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DIVIDE_BY_ZERO,
                reduce_dims=["x", "y"],
                both_zero=False,
            ),
        ],
        "test_nse_no_error_no_warn": [
            # test no options
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                nse_kwargs={},
                expect_dims=[],
                expect_shape=(1,),
            ),
            # test multiple options
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                nse_kwargs=dict(
                    weights=np.abs(_OBS_DEFAULT),
                    reduce_dims=None,
                    preserve_dims=["x", "y"],
                    is_angular=False,
                ),
                expect_dims=["x", "y"],
                expect_shape=(2, 3),
            ),
            # test angular
            dict(
                fcst=_FCST_DEFAULT * 360,
                obs=_OBS_DEFAULT * 360,
                nse_kwargs=dict(reduce_dims="t", is_angular=True),
                expect_dims=["x", "y"],
                expect_shape=(2, 3),
            ),
        ],
        "test_nse_nan_broadcasting": [
            dict(
                fcst=xr.DataArray([3, 4, 5, 6, 7]),
                obs=xr.DataArray([2, 3, 4, 5, 6]),
                expect_nse=0.5,
            ),
            dict(
                fcst=xr.DataArray([np.nan, 4, 5, 6, 7]),
                obs=xr.DataArray([2, 3, 4, 5, 6]),
                expect_nse=0.2,
            ),
            dict(
                fcst=xr.DataArray([3, 4, 5, 6, 7]),
                obs=xr.DataArray([np.nan, 3, 4, 5, 6]),
                expect_nse=0.2,
            ),
            dict(
                fcst=xr.DataArray([np.nan, 4, 5, 6, 7]),
                obs=xr.DataArray([np.nan, 3, 4, 5, 6]),
                expect_nse=0.2,
            ),
        ],
    }

    def test_error_incompatible_dims(
        self,
        fcst,
        obs,
        reduce_dims,
        preserve_dims,
        expect_context,
    ):
        """
        Tests dimension incompatibility raises errors
        """
        with expect_context:
            nse(
                fcst,
                obs,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
            )

    def test_error_insufficient_data(self, fcst, obs, reduce_dims):
        """
        Should raise DimensionError if the theres only one item to be reduced, as
        this cannot be used to compute the observation variance (=0 with one item,
        and guarenteed to cause every field to divide by zero), this means the score
        will not produce anything meaningful - so an error is thrown to show this is
        the case.
        """
        with pytest.raises(ValueError):
            nse(fcst, obs, reduce_dims=reduce_dims)

    def test_error_invalid_weights(self, fcst, obs, weights, reduce_dims):
        """
        Should raise an error if weights:
            - contain a negative element, and the following cases raise errors in case of
              unintentional inputs.
            - are all nans (everything is masked - nothing to compute)
            - are all zeros (everything is zero forced - score is NaN)
        """
        with pytest.raises(ValueError):
            nse(
                fcst,
                obs,
                weights=weights,
                reduce_dims=reduce_dims,
            )

    def test_warn_divide_by_zero(self, fcst, obs, reduce_dims, both_zero):
        """
        Should warn when divide by zero error happens, but not raise an error, tests two cases:
            - when both obs and fcst are 0 - should have a NaN result
            - when only obs is 0 - should have -Inf in the result
        """
        # should have one fabricated -Inf entry at t=1
        with pytest.warns(RuntimeWarning):
            ret = nse(
                fcst,
                obs,
                reduce_dims=reduce_dims,
            )
            if both_zero:
                assert np.any(np.isnan(ret[1]))
            else:
                assert np.any(np.isneginf(ret[1]))

    def test_nse_no_error_no_warn(self, fcst, obs, nse_kwargs, expect_dims, expect_shape):
        """
        Tests the typical behaviour of NSE with some different argument combinations
        - should not raise any warnings or errors.
        """
        ret = nse(fcst, obs, **nse_kwargs)
        assert np.all(ret <= 1.0)
        assert isinstance(ret, xr.DataArray)
        assert ret.name == "NSE"
        if len(expect_dims) > 0:
            assert ret.shape == expect_shape
            assert all(d in ret.dims for d in expect_dims)

    def test_nse_nan_broadcasting(self, fcst, obs, expect_nse):
        """
        Tests that nans are matched properly between fcst and obs.
        """
        res = nse(fcst, obs)
        assert np.abs(res - expect_nse) <= 1e-6


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
        expected behaviour:
        - reduce_dims must be specified such that the dimensions being reduced exist in both arrays
        - the result can then be broadcast to the remaining dimensions appropriately
        - tapioca is ignored i.e. variables that do not exist in both datasets
        - no raised errors are tested here as they should be handled by utility calls
        """
        ds_obs = xr.Dataset(
            data_vars=dict(
                temp=NseSetup.make_random_xr_array((3, 5, 2), ["x", "y", "t"]),
                precip=NseSetup.make_random_xr_array((3, 5, 2), ["x", "y", "t"]),
            ),
        )
        ds_fcst = xr.Dataset(
            data_vars=dict(
                temp=NseSetup.make_random_xr_array((3, 5, 2, 4), ["x", "y", "t", "h"]),
                precip=NseSetup.make_random_xr_array((3, 5), ["x", "y"]),
                tapioca=NseSetup.make_random_xr_array((2, 5), ["t", "y"]),
            ),
        )
        reduce_dims = ["x", "y"]
        res = nse(ds_fcst, ds_obs, reduce_dims=reduce_dims)
        # result is a dataset
        assert isinstance(res, xr.Dataset)
        # variables are data arrays
        assert isinstance(res["precip"], xr.DataArray)
        assert isinstance(res["temp"], xr.DataArray)
        # precip should only have "t", since "h" isn't defined for either obs or fcst in precip
        # HOWEVER, because of broadcasting, nan values get added in to match temp.
        assert set(res["precip"].dims) == set(["t", "h"])
        assert res["precip"].shape == (2, 4)
        # temp should have both "t" and "h"
        assert set(res["temp"].dims) == set(["t", "h"])
        assert res["temp"].shape == (2, 4)
        # tapioca is ignored
        assert "tapioca" not in res.data_vars.keys()


class TestNseDask(NseSetup):
    """
    Basic testing if dask is available and used appropriately by NSE.

    NOTE: failure conditions will not be the responsibility of this test, this suite just exists to
    check if dask computes things appropriately with non-dask as a compatiblity measure.
    """

    pytestmark = pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask unavailable, could not run test")

    def test_nse_with_dask_inputs(self, tmpdir):
        """
        Basic test to see if NSE works with dask. This is a contrived setup, and we're just looking
        at whether compatiblity exists.

        Detailed analysis is currently out of scope.
        """
        # prep dataarrays - probably not very optimal chunk strategy
        chunks = {"x": 25, "y": 25}
        da1 = self.make_random_xr_array((100, 100, 10), ("x", "y", "t")).chunk(chunks)
        da2 = da1 * 0.99  # make them almost equal - [1]

        res = nse(da1, da2, reduce_dims=("x", "y"))
        assert dask.is_dask_collection(res)  # SHOULD return a dask array if chunked

        # Load into memory and perform computation
        true_res = res.compute()

        # SHOULD be a regular DataArray after compute()
        assert isinstance(true_res, xr.DataArray)
        # SHOULD be close to 1 ~= NSE >> 0 see: [1]
        # using "any" instead of "all" as a weak check, so this is unlikely to fail
        not_terrible = (true_res > 0).any().item()
        not_wrong = (true_res <= 1).all().item()
        # SHOULD NOT be dask anymore, typecheck: bool
        assert isinstance(not_terrible and not_wrong, bool)
        # Do the actual assertion for [1]
        assert not_terrible and not_wrong


## for KGE
DA1_KGE = xr.DataArray(
    np.array([[1, 2, 3], [0, 1, 0], [0.5, -0.5, 0.5], [3, 6, 3]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)

DA2_KGE = xr.DataArray(
    np.array([[2, 4, 6], [6, 5, 6], [3, 4, 5], [3, np.nan, 3]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)

DA3_KGE = xr.DataArray(
    np.array([[1, 2, 3], [3, 2.5, 3], [1.5, 2, 2.5], [1.5, np.nan, 1.5]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)
DA4_KGE = xr.DataArray(
    np.array([[1, 3, 7], [2, 2, 8], [3, 1, 7]]),
    dims=("space", "time"),
    coords=[
        ("space", ["x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)
DA5_KGE = xr.DataArray(
    np.array([1, 2, 3]),
    dims=("space"),
    coords=[("space", ["x", "y", "z"])],
)

## Expected KGE values
EXP_KGE_KEEP_SPACE_DIM = xr.DataArray(
    np.array([0.2928932188134524, -1.2103875562418747, -0.44811448882050064, np.nan]),
    dims=("space"),
    coords=[("space", ["w", "x", "y", "z"])],
)
EXP_KGE_REDUCE_ALL = xr.DataArray(0.2928932188134524)
EXP_KGE_REDUCE_ALL_MODIFIED = xr.DataArray(0.5)

EXP_KGE_rho_returns_components = xr.DataArray(1.0)
EXP_KGE_alpha_returns_components = xr.DataArray(0.5)
EXP_KGE_gamma_returns_components = xr.DataArray(1.0)
EXP_KGE_beta_returns_components = xr.DataArray(0.5)

EXP_KGE_returns_components = xr.Dataset(
    {
        "kge": EXP_KGE_REDUCE_ALL,
        "rho": EXP_KGE_rho_returns_components,
        "alpha": EXP_KGE_alpha_returns_components,
        "beta": EXP_KGE_beta_returns_components,
    }
)

EXP_KGE_returns_components_modified = xr.Dataset(
    {
        "kge": EXP_KGE_REDUCE_ALL_MODIFIED,
        "rho": EXP_KGE_rho_returns_components,
        "gamma": EXP_KGE_gamma_returns_components,
        "beta": EXP_KGE_beta_returns_components,
    }
)


EXP_KGE_Scaling_Factors = xr.DataArray(
    1 - np.sqrt((0.5 * (1 - 1)) ** 2 + (1.0 * (0.5 - 1)) ** 2 + (2 * (0.5 - 1)) ** 2)
)


EXP_KGE_DIFF_SIZE = xr.DataArray(
    np.array([1.0, -1.0, -1.8791915368841288]),
    dims=("time"),
    coords=[("time", [1, 2, 3])],
)

## Parametrized test for kge function to check various incorrect types and sizes
Incorrect_Input_KGE = xr.Dataset(
    data_vars={
        "temperature": ("x", [10, 20, 30]),
    },
    coords={
        "x": [0, 1, 2],
    },
)
Incorrect_SFactors_Type_KGE = "incorrect_type"
Incorrect_SFactors_List_KGE = [1, 2]
Incorrect_SFactors_Numpy_KGE = np.array([1, 2, 3, 4])

EXP_KGE_message1 = "kge: fcst must be an xarray.DataArray"
EXP_KGE_message2 = "kge: obs must be an xarray.DataArray"
EXP_KGE_message3 = "kge: scaling_factors must be an iterable of exactly 3 elements"
EXP_KGE_message4 = "kge: method must be either '2009' or '2012'"


@pytest.mark.parametrize(
    ("fcst", "obs", "reduce_dims", "preserve_dims", "include_components", "scaling_factors", "method", "expected"),
    [
        # Check reduce dim arg
        (DA1_KGE, DA2_KGE, None, "space", False, None, "2009", EXP_KGE_KEEP_SPACE_DIM),
        # Check preserve dim arg
        (DA1_KGE, DA2_KGE, "time", None, False, None, "2009", EXP_KGE_KEEP_SPACE_DIM),
        # Check reduce all
        (DA3_KGE, DA2_KGE, None, None, False, None, "2009", EXP_KGE_REDUCE_ALL),
        # returning components
        (DA3_KGE, DA2_KGE, None, None, True, None, "2009", EXP_KGE_returns_components),
        # Check scaling_factors
        (DA3_KGE, DA2_KGE, None, None, False, [0.5, 1.0, 2.0], "2009", EXP_KGE_Scaling_Factors),
        # Check different size arrays as input
        (DA4_KGE, DA5_KGE, "space", None, False, None, "2009", EXP_KGE_DIFF_SIZE),
        # Check method arguments
        (DA3_KGE, DA2_KGE, None, None, True, None, "2012", EXP_KGE_returns_components_modified),
    ],
)
def test_kge(fcst, obs, reduce_dims, preserve_dims, include_components, scaling_factors, method, expected):
    """
    Tests continuous.kge
    """
    result = kge(
        fcst,
        obs,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        include_components=include_components,
        scaling_factors=scaling_factors,
        method=method,
    )
    xr.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_kge_dask():
    """
    Tests that continuous.kge works with Dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    fcst = DA3_KGE.chunk()
    obs = DA2_KGE.chunk()
    result = kge(fcst, obs)
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_equal(result, EXP_KGE_REDUCE_ALL)


@pytest.mark.parametrize(
    "fcst, obs, scaling_factors, method, expected_exception, expected_message",
    [
        # Test case for fcst with incorrect type (list instead of xr.DataArray)
        (Incorrect_Input_KGE, DA2_KGE, None, "2009", TypeError, EXP_KGE_message1),
        # Test case for obs with incorrect type (list instead of xr.DataArray)
        (DA1_KGE, Incorrect_Input_KGE, None, "2009", TypeError, EXP_KGE_message2),
        # Test case for scaling_factors with incorrect type (string instead of list or np.ndarray)
        (DA1_KGE, DA2_KGE, Incorrect_SFactors_Type_KGE, "2009", ValueError, EXP_KGE_message3),
        # Test case for scaling_factors with incorrect number of elements (list with 2 elements)
        (DA1_KGE, DA2_KGE, Incorrect_SFactors_List_KGE, "2009", ValueError, EXP_KGE_message3),
        # Test case for scaling_factors with incorrect number of elements (numpy array with 4 elements)
        (DA1_KGE, DA2_KGE, Incorrect_SFactors_Numpy_KGE, "2009", ValueError, EXP_KGE_message3),
        # Test case for method with incorrect value (not 'original' or 'modified')
        (DA1_KGE, DA2_KGE, Incorrect_SFactors_Numpy_KGE, "invalid_method", ValueError, EXP_KGE_message4),
    ],
)
def test_kge_errors(fcst, obs, scaling_factors, method, expected_exception, expected_message):
    """
    Test continuous.kge raises error with an incorrect type and sizes
    """
    with pytest.raises(expected_exception, match=expected_message):
        kge(fcst, obs, scaling_factors=scaling_factors, method=method)


## for pbias
EXP_PBIAS1 = xr.DataArray(
    np.array([-50, -100.0, (0.5 / 3 + 0.5 / 3) / (-0.5 / 3) * 100]),
    dims=("space"),
    coords=[
        ("space", ["w", "x", "y"]),
    ],
)
EXP_PBIAS2 = xr.DataArray(
    np.array([100.0, np.inf, (0.5 / 3 + 0.5 / 3) / (-0.5 / 3) * 100]),
    dims=("space"),
    coords=[
        ("space", ["w", "x", "y"]),
    ],
)

EXP_PBIAS3 = xr.DataArray(
    np.array([-50.0, -100.0, -75.0]),
    dims=("space"),
    coords=[
        ("space", ["w", "x", "y"]),
    ],
)

EXP_PBIAS4 = xr.DataArray(np.array(-13 / 15.5 * 100))

EXP_DS_PBIAS1 = xr.Dataset({"a": EXP_PBIAS1, "b": EXP_PBIAS2})


@pytest.mark.parametrize(
    ("fcst", "obs", "reduce_dims", "preserve_dims", "weights", "expected"),
    [
        # Check reduce dim arg
        (DA1_BIAS, DA2_BIAS, None, "space", None, EXP_PBIAS1),
        # Check divide by zero returns a np.inf
        (DA2_BIAS, DA1_BIAS, None, "space", None, EXP_PBIAS2),
        # Check weighting works
        (DA1_BIAS, DA3_BIAS, None, "space", BIAS_WEIGHTS, EXP_PBIAS3),
        # # Check preserve dim arg
        (DA1_BIAS, DA2_BIAS, "time", None, None, EXP_PBIAS1),
        # Reduce all
        (DA1_BIAS, DA2_BIAS, None, None, None, EXP_PBIAS4),
        # Test with Dataset
        (DS_BIAS1, DS_BIAS2, None, "space", None, EXP_DS_PBIAS1),
    ],
)
def test_pbias(fcst, obs, reduce_dims, preserve_dims, weights, expected):
    """
    Tests continuous.pbias
    """
    result = pbias(fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights)
    # xr.testing.assert_equal(result, expected)
    xr.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_pbias_dask():
    """
    Tests that continuous.pbias works with Dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    fcst = DA1_BIAS.chunk()
    obs = DA3_BIAS.chunk()
    weights = BIAS_WEIGHTS.chunk()
    result = pbias(fcst, obs, preserve_dims="space", weights=weights)
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_equal(result, EXP_PBIAS3)
