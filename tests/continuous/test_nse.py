# disable non-critical pylinting for tests:
# pylint: disable=use-dict-literal
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-few-public-methods
# pylint: disable=protected-access
"""
Collection of tests for the NSE score, contains:
    - NseSetup: Base class used by all other test classes
    - TestNsePublicApi: Tests concerning the public API interface - this is the main suite
    - TestNseInternals: Tests specific to the utils that are not covered by public API
    - TestNseScore: Tests specific to score computation not already covered by public API
    - TestNseDataset: Tests compatiblity with datasets (most tests use data array for convenience)
    - TestNseDask: Tests compatibility with dask
"""
import os
import typing

import numpy as np
import pytest
import xarray as xr
from numpy import typing as npt

import scores.continuous.nse_impl as nse_impl
from scores.utils import DimensionError

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
                expect_context=pytest.raises(DimensionError),
            ),
            # preserve all (explicitly specified)
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                reduce_dims=None,
                preserve_dims=["x", "y", "t"],
                expect_context=pytest.raises(DimensionError),
            ),
            # no dims reduced - essentially the same as preserve all
            dict(
                fcst=_FCST_DEFAULT,
                obs=_OBS_DEFAULT,
                reduce_dims=[],
                preserve_dims=None,
                expect_context=pytest.raises(DimensionError),
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
            nse_impl.nse(
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
        with pytest.raises(DimensionError):
            nse_impl.nse(fcst, obs, reduce_dims=reduce_dims)

    def test_error_invalid_weights(self, fcst, obs, weights, reduce_dims):
        """
        Should raise an error if weights:
            - contain a negative element, and the following cases raise errors in case of
              unintentional inputs.
            - are all nans (everything is masked - nothing to compute)
            - are all zeros (everything is zero forced - score is NaN)
        """
        with pytest.raises(ValueError):
            nse_impl.nse(
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
        with pytest.warns(UserWarning):
            ret = nse_impl.nse(
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
        ret = nse_impl.nse(fcst, obs, **nse_kwargs)
        assert np.all(ret <= 1.0)
        assert isinstance(ret, xr.DataArray)
        assert ret.name == "NSE"
        if len(expect_dims) > 0:
            assert ret.shape == expect_shape
            assert all(d in ret.dims for d in expect_dims)


class TestNseInternals(NseSetup):
    """
    NOTE: most of NseUtils is tested by the public API test suite and is not repeated here.
    Only things missed by the public API test suite will be covered here.
    """

    META_INPUT_DA = nse_impl.NseMetaInput(
        datasets=[],  # type: ignore
        gathered_dims=[],  # type: ignore
        is_angular=False,  # doesn't matter
        is_dataarray=True,  # >>> checking for this
    )

    META_INPUT_NOT_DA = nse_impl.NseMetaInput(
        datasets=[],  # type: ignore
        gathered_dims=[],  # type: ignore
        is_angular=False,  # doesn't matter
        is_dataarray=False,  # >>> checking for this
    )

    params = {
        "test_try_extract_singleton_dataarray": [
            # empty - should not return anything
            dict(ds=xr.Dataset(), expect_da=None),
            # multiple dataarrays should not return anything
            dict(
                ds=xr.Dataset(dict(x=xr.DataArray([1]), y=xr.DataArray([1]))),
                expect_da=None,
            ),
            # single dataarrays should return:
            # Using fully specified array - to check that dims/coordinates and
            # metadata do not affect extraction of ``keys``
            # i.e. only FOO should count as a key
            dict(
                ds=xr.Dataset(
                    data_vars=dict(FOO=xr.DataArray([1, 2, 3, 4, 5], dims="x")),
                    coords={"x": ["I", "AM", "NOT", "A", "KEY"]},
                    attrs=dict(description="I should not be counted as a key either"),
                ),
                expect_da=xr.DataArray(
                    [1, 2, 3, 4, 5],
                    name="FOO",
                    coords={"x": ["I", "AM", "NOT", "A", "KEY"]},
                ),
            ),
        ],
        # unparametrized
        "test_check_all_same_type_error": [{}],
        # unparametrized
        "test_check_all_same_type_none_ignored": [{}],
        "test_check_metadata_consistency_error": [
            # is_dataarray do not match
            dict(
                meta_score=nse_impl.NseMetaScore(
                    components=nse_impl.NseComponents(
                        fcst_error=xr.Dataset(dict(x=xr.DataArray([1]))),
                        obs_variance=xr.Dataset(dict(x=xr.DataArray([1]))),
                        # give it a value in case nse check happens first:
                        nse=xr.Dataset(dict(x=xr.DataArray([1]))),
                    ),
                    ref_meta_input=META_INPUT_NOT_DA,  # >>> checking this
                ),
                meta_input=META_INPUT_DA,  # >>> checking this
            ),
            # is_dataarray do not match - flipped
            dict(
                meta_score=nse_impl.NseMetaScore(
                    components=nse_impl.NseComponents(
                        fcst_error=xr.Dataset(dict(x=xr.DataArray([1]))),
                        obs_variance=xr.Dataset(dict(x=xr.DataArray([1]))),
                        # give it a valid value in case nse check happens first:
                        nse=xr.Dataset(dict(x=xr.DataArray([1]))),
                    ),
                    ref_meta_input=META_INPUT_DA,  # >>> checking this
                ),
                meta_input=META_INPUT_NOT_DA,  # >>> checking this
            ),
            # is data array but has multiple keys
            dict(
                meta_score=nse_impl.NseMetaScore(
                    components=nse_impl.NseComponents(
                        fcst_error=xr.Dataset(dict(x=xr.DataArray([1]))),
                        obs_variance=xr.Dataset(dict(x=xr.DataArray([1]))),
                        # >>> checking for this: multiple keys
                        nse=xr.Dataset(dict(x=xr.DataArray([1]), y=xr.DataArray([2]))),
                    ),
                    ref_meta_input=META_INPUT_DA,
                ),
                meta_input=META_INPUT_DA,
            ),
            # is data array but has no keys
            dict(
                meta_score=nse_impl.NseMetaScore(
                    components=nse_impl.NseComponents(
                        fcst_error=xr.Dataset(dict(x=xr.DataArray([1]))),
                        obs_variance=xr.Dataset(dict(x=xr.DataArray([1]))),
                        # >>> checking for this: empty
                        nse=xr.Dataset(),
                    ),
                    ref_meta_input=META_INPUT_DA,
                ),
                meta_input=META_INPUT_DA,
            ),
        ],
        "test_check_metadata_consistency_okay": [
            # both datasets - nothing to check/not possible to detect errors
            dict(
                meta_score=nse_impl.NseMetaScore(
                    components=nse_impl.NseComponents(
                        fcst_error=xr.Dataset(dict(x=xr.DataArray([1]))),
                        obs_variance=xr.Dataset(dict(x=xr.DataArray([1]))),
                        # give it a value in case nse check happens first:
                        nse=xr.Dataset(dict(x=xr.DataArray([1]))),
                    ),
                    ref_meta_input=META_INPUT_NOT_DA,
                ),
                meta_input=META_INPUT_NOT_DA,
            ),
            # both is_dataarray and score only has one key
            dict(
                meta_score=nse_impl.NseMetaScore(
                    components=nse_impl.NseComponents(
                        fcst_error=xr.Dataset(dict(x=xr.DataArray([1]))),
                        obs_variance=xr.Dataset(dict(x=xr.DataArray([1]))),
                        # >>> checking for this:
                        nse=xr.Dataset(dict(x=xr.DataArray([1]))),
                    ),
                    ref_meta_input=META_INPUT_DA,
                ),
                meta_input=META_INPUT_DA,
            ),
        ],
    }

    def test_try_extract_singleton_dataarray(self, ds, expect_da):
        """
        Try to extract from dataset with multiple keys or no keys - should trigger an
        error.
        """
        maybe_da = nse_impl.NseUtils.try_extract_singleton_dataarray(ds)

        if expect_da is None:
            assert maybe_da is None
        else:
            assert isinstance(maybe_da, xr.DataArray)
            assert maybe_da.identical(expect_da)

    def test_check_all_same_type_error(self):
        """
        Checks for raising error condition when checking all same type.
        The success & failure conditions are already checked by ``tests_typing.py``
        """
        with pytest.raises(TypeError):
            nse_impl.NseUtils.check_all_same_type(
                xr.DataArray([1]),
                xr.Dataset(dict(y=xr.DataArray([1]))),
            )

    def test_check_all_same_type_none_ignored(self):
        """
        Optionals are allowed in this check
        """
        nse_impl.NseUtils.check_all_same_type(
            xr.DataArray([1]),
            None,
        )

    def test_check_metadata_consistency_error(self, meta_score, meta_input):
        """
        Fail condition
            - One is a datarray but other isn't
            - Both are datasets but score has either no keys or multiple keys
        """
        with pytest.raises(RuntimeError):
            nse_impl.NseUtils.check_metadata_consistency(meta_score, meta_input)

    def test_check_metadata_consistency_okay(self, meta_input, meta_score):
        """
        Success condition
            - both are datasets or
            - both are dataarrays and score as single key
        """
        nse_impl.NseUtils.check_metadata_consistency(meta_score, meta_input)


class TestNseScore(NseSetup):
    """
    Tests validity of the actual NSE scoring functions for a given set of inputs. As such, tests
    here should not use ``np.random.random``, instead they'd need to be either handcrafted or
    verifiable by a naive secondary algorithm.
    """

    @classmethod
    def setup_class(cls):
        """
        Worked example - note this is not a scientific example, its still very much contrived.

        However, it is intentionally made deterministic to test the intuitive outcomes of the
        underlying (decomposed) mathematical calculations.

        This is very useful for verifying the computational soundness NSE, particularly by
        comparing it against intuitive handwritten solutions which are representated as fractions,
        rather than randomly generated values.

        Further it also compares against a naive implementation of NSE e.g. using multi-indexed for
        loops (essentially same as nested for loops but with numpy so that it looks less cluttered -
        see: ``test_nse_against_naive_impl``)

        ------------------------------------------------------------------------------------------
         setup
        ------------------------------------------------------------------------------------------
            arbitrarily using a dataset, since most of the public API tests uses a data array

            dims:    2 * 4 * 2 * 3    : x, y, t, l => 48 values
            obs:     1, 2, 3, 4...    reshaped
            fcst:    2, 3, 4, 5...    reshaped
            weights: ...?             to be introduced below

        ------------------------------------------------------------------------------------------
         obs mean calculation - scratch work
        ------------------------------------------------------------------------------------------
            reduce over (x, y) => 1 nse result per (t, l) 2 x 3 => 6 elements being reduced
            => (2i + 5) / 2 is the mean, where i is the starting index of each group,
               assuming (t, l) are the innermost elements of the whole data structure.
            => e.g. for i = 1  | obs = 1, 2, 3, 4, 5 ,6,
                               | mean = 3 + 4 / 2 = 3.5 = (2*1 + 5)/2
            => e.g. for i = 10 | obs = 10, 11, 12, 13, 14, 15
                               | mean = 12 + 13 / 2 = 12.5 = (2*10 + 5)/2

        ------------------------------------------------------------------------------------------
         forecast error calculation
        ------------------------------------------------------------------------------------------
            since the error is always 1, the square is 1, and hence

            +---------------------------------------------+
            | forecast error = (1^2) * 6 / 6 = 1  --> [1] |
            +---------------------------------------------+

            probably not very surprising.

        ------------------------------------------------------------------------------------------
         obs variance calculation
        ------------------------------------------------------------------------------------------
            intuition: the variance should be the same since the "spread" of data for each group
            being reduced should be the same (again assuming t, l are the inner most indices).

            given any starting index, mean = (2i + 5)/2, obs = j: i->i+5 (total 6), so...

            error_j = (j - (2i + 5)/2)^2: j from i -> i+5

            because j is always relative to i, the first error term is -5/2 and the last error term
            is 5/2. We can also assume because i increases linearly, the error term must also
            increase linearly from -5/2 -> 5/2, since there are 6 entries there are 5 intervals.

            (5/2-(-5/2))/5 = 10/10 = 1, which means the errors also increase by 1

            obs_variance = (-5/2)^2 + (-3/2)^2 + (-1/2)^2  + (1/2)^2 + (3/2)^2 + (5/2)^2 / 6
                         = (25 + 9 + 1 + 1 + 9 + 25) / (6 * 4) = 70 / 24 ~= 2.9166... ---> [2]

            +------------------------------------------------------------------+
            | obs variance = sum_j((j-i)+5/2)/6 ~= 70 / 24 ~= 2.9166... -> [2] |
            +------------------------------------------------------------------+

        ------------------------------------------------------------------------------------------
         NSE without weights
        ------------------------------------------------------------------------------------------
            combining [1] and [2] we get NSE without weights

            +--------------------------------------------------+
            | NSE = 1 - 24 / 70 = 46 / 70 ~= 0.657... ---> [3] |
            +--------------------------------------------------+

            this makes sense the obs groups have a differences uniformly ranging from 0 to 5,
            whereas the fcst is always bounded to 1, which means it is a better predictor than the
            mean. If the (t, l) is grouped according to the assumption then for every (x, y) element
            should have the same value: 46/70 ~= 0.657

        ------------------------------------------------------------------------------------------
         NSE with weights
        ------------------------------------------------------------------------------------------
            Suppose we want to add some weights - say we want to weight lead times (l) now suppose
            we want a triangular weighting scheme because why not, [0,3,1]. This also forces the
            first entry to 0 for <insert arbitrary reason here>.

            Broadcasting this over time (t), the full weighting is [[0,3,1],[0,3,1]].

            So now essentially we have (noting that weights are applied to the squared error):
              - fcst error   = (0+3+1+0+3+1)/6 = 8/6 = 4/3 = 1.333...
              - obs variance = ((9/4)*3 + (1/4)*1 + (9/4)*3 + (25/4)*1)/6
                             = (27+1+27+25)/24 = 80/24 = 10/3 = 3.333...

            +------------------------------------------------------------+
            | NSE with weights = 1 - 24*4/3*80 = 3/5 = 0.6... ---> [4a]  |
            +------------------------------------------------------------+

            Note that the error has not changed much, but if we think about it we have 0s in
            (unintentionally) strategic spots, the first 0 eliminates a large error term in obs,
            while the second 0 elements a small error term, this sort of cancels each other out.

            However, this will still have the same value for all the elements in the returned array.
            If we were to vary things a little, we could also add weights based on (x, y)
            coordinates. Because (x, y) are not being reduced - this is essentially a scaling
            operation.

            Arbitrarily lets just scale on x which has cardinality=2. If we choose [1,2] as the
            weighting on x then effectively we have for:
                - x = 0: weights along l = [0,3,1] => we already computed this
                - x = 1: weights along l = [0,6,3]
                - since there is a slight assymetry in the error, we'd expect x=1 to have a slightly
                  different score, since its not quite double
            for x = 1:
                - fcst error   = 18/6  = 3.0
                - obs variance = (6*9+3*1+6*9+3*25)/24 = 186/24 => 7.75

            +---------------------------------------------------------------+
            | NSE with weights = 1 - 72/186 = 114/186 = 0.612... ---> [4b]  |
            +---------------------------------------------------------------+

            So the result will be for x=0: 0.6 and x=1: 0.612...

        ------------------------------------------------------------------------------------------
         Let's add another variable - it is a dataset after all
        ------------------------------------------------------------------------------------------
            Let's assume the above was for temperature, let's make our life harder and add another
            variable - precipitation. This time let's make the difference cycle between 2 and 0
            instead of 1 the average error is still 1 - but let's see if it affects the score

            i.e. fcst = [obs+2, obs+0, obs+2, obs+0, ...]

            +-------------------------------------------------------------------------+
            | now:                                                                    |
            |     - forecast error = (2^2 * 3) / 6 = 2: shock! error has increased!   |
            |     - obs variance = still the same (thankfully)                        |
            |     - nse without weights = 1 - (2 * 24 / 70)                           |
            |                           = 22 / 70 ~= 0.314 (worse than before)        |
            |     --> [6a,6b,6c]                                                      |
            +-------------------------------------------------------------------------+

            We're slighly worse off - note we have effectively made our predictions less
            consistent/reliable and as a result more noisy even when compared to the obs. In fact
            the spread ratio of error:variance is 2:2.91, whereas before it was 1:2.91
            previously. Eventhough, our _average_ error has not changed!

            lets add some cheeky weights now to conveniently only select the forecasts with 0
            errors. This time the result should be easy to work out i.e. if we have no error the
            best score NSE can give us is 1.0.

            How do we choose the weights? We use the selection pattern i.e. 0 if index is even, 1 if
            odd, assuming assuming the odd fcst indices match the obs and the evens are 2 away from
            the obs. Set it to [[0, 1, 0], [2, 0, 2]], arbitrarily, the value doesn't matter, as
            long as it matches the parts where forecast error = 0.

            (note: with weights the obs error is 61/24 - though this won't matter.)

            +--------------------------------------+
            | NSE with weights = 1.0 (QED) -> [7]  |
            +--------------------------------------+

        ------------------------------------------------------------------------------------------
        .. note::

            All of the above was done by hand so that it can be analytically verifiable against
            fractional results, derived using some heuristic shortcuts.

            However, we also compare it with ``nse_naive`` helper in NSE setup, which uses a
            bunch of nested for loops to do the same effective computation.
        """
        # -----------------------------------------------
        #  dimensions | x:2,y:4,t:2,l:3: total:48 values
        # -----------------------------------------------
        # -> reducing over (t, l)

        def _build_ds(np_temp, np_precip):
            _dims = ["x", "y", "t", "l"]
            return xr.Dataset(
                dict(
                    temperature=xr.DataArray(np_temp, dims=_dims),
                    precipitation=xr.DataArray(np_precip, dims=_dims),
                )
            )

        # build temperature dataarray - offset by 1:
        temp_obs = np.linspace(start=0, stop=48, num=48, dtype=int, endpoint=False)
        temp_fcst = np.linspace(start=1, stop=49, num=48, dtype=int, endpoint=False)
        temp_obs = np.reshape(temp_obs, (2, 4, 2, 3))
        temp_fcst = np.reshape(temp_fcst, (2, 4, 2, 3))

        # build precipitation dataarray - offset by 2 on even entries:
        # actual start value shouldn't actually matter since the error is relative
        precip_obs = np.linspace(start=10, stop=58, num=48, dtype=int, endpoint=False)
        precip_fcst = np.linspace(start=10, stop=58, num=48, dtype=int, endpoint=False)
        precip_fcst[slice(0, None, 2)] += 2  # offset every even index
        precip_obs = np.reshape(precip_obs, (2, 4, 2, 3))
        precip_fcst = np.reshape(precip_fcst, (2, 4, 2, 3))

        # weights - unbroadcasted
        temp_weights = np.array([[[[0, 3, 1]]], [[[0, 6, 3]]]])  # x,.,.,l: 2*1*1*3
        precip_weights = np.array([[[[0, 1, 0], [2, 0, 2]]]])  # .,.,t,l: 1*1*2*3

        # helper to convert to expected output - this will always be x=2 * y=4,
        # because we're reducing by (t, l)
        def _build_exp_output_da(_x, name, dim_names=("x", "y")):
            return xr.DataArray(
                np.reshape(np.repeat(_x, repeats=8), (2, 4)),
                dims=dim_names,
                name=name,
            )

        # -------------------------
        #  Prepare expected values
        # -------------------------
        # See docstrings of this setup function to see how they were derived

        # --- without weights ---
        exp_temp_fcst_err = _build_exp_output_da(1.0, "temperature")
        exp_temp_obs_var = _build_exp_output_da(70.0 / 24.0, "temperature")
        exp_temp_nse_scr = _build_exp_output_da(46.0 / 70.0, "temperature")
        exp_precip_fcst_err = _build_exp_output_da(2.0, "precipitation")
        exp_precip_obs_var = _build_exp_output_da(70.0 / 24.0, "precipitation")
        exp_precip_nse_scr = _build_exp_output_da(22.0 / 70.0, "precipitation")

        # --- with weights ---
        # nomenclature [t|p][e|v|s][1|2]
        # t = temperature, p = precipitation
        # e = error, v = variance, s = score
        # 1 = first weight, 2 = second weight
        # - NOTE:
        #    we will have two different values for weights, repeated at specific positions because
        #    of the way it is reduced.

        # temperature
        te1 = 4.0 / 3.0
        tv1 = 10.0 / 3.0
        ts1 = 3.0 / 5.0
        te2 = 3.0
        tv2 = 186.0 / 24.0
        ts2 = 114.0 / 186.0

        # result depends on x axis, and must be 2*4
        exp_temp_fcst_err_weights = xr.DataArray(
            np.array([[te1] * 4, [te2] * 4]),
            dims=["x", "y"],
        )
        exp_temp_obs_var_weights = xr.DataArray(
            np.array([[tv1] * 4, [tv2] * 4]),
            dims=["x", "y"],
        )
        exp_temp_nse_scr_weights = xr.DataArray(
            np.array([[ts1] * 4, [ts2] * 4]),
            dims=["x", "y"],
        )

        # precipitation - note precipitation only has one value for weights
        pe1 = 0.0
        pv1 = 61.0 / 24.0
        ps1 = 1.0
        exp_precip_fcst_err_weights = _build_exp_output_da(pe1, "precipitation")
        exp_precip_obs_var_weights = _build_exp_output_da(pv1, "precipitation")
        exp_precip_nse_scr_weights = _build_exp_output_da(ps1, "precipitation")

        # expose expected values by assigning them to the class
        # --- no weights ---
        cls.exp_fcst_error = xr.Dataset(
            dict(
                temperature=exp_temp_fcst_err,
                precipitation=exp_precip_fcst_err,
            )
        )
        cls.exp_obs_variance = xr.Dataset(
            dict(
                temperature=exp_temp_obs_var,
                precipitation=exp_precip_obs_var,
            )
        )
        cls.exp_nse_score = xr.Dataset(
            dict(
                temperature=exp_temp_nse_scr,
                precipitation=exp_precip_nse_scr,
            )
        )

        # --- with weights ---
        cls.exp_fcst_error_weights = xr.Dataset(
            dict(
                temperature=exp_temp_fcst_err_weights,
                precipitation=exp_precip_fcst_err_weights,
            )
        )
        cls.exp_obs_variance_weights = xr.Dataset(
            dict(
                temperature=exp_temp_obs_var_weights,
                precipitation=exp_precip_obs_var_weights,
            )
        )
        cls.exp_nse_score_weights = xr.Dataset(
            dict(
                temperature=exp_temp_nse_scr_weights,
                precipitation=exp_precip_nse_scr_weights,
            )
        )

        # --------
        #  inputs
        # --------
        # expose input values by assigning them to the class
        cls.ds_obs = _build_ds(temp_obs, precip_obs)
        cls.ds_fcst = _build_ds(temp_fcst, precip_fcst)
        expand_dims = {"x": 2, "y": 4, "t": 2, "l": 3}
        cls.ds_weights = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.broadcast_to(temp_weights, expand_dims.values()),
                    dims=expand_dims.keys(),
                ),
                "precipitation": xr.DataArray(
                    np.broadcast_to(precip_weights, expand_dims.values()),
                    dims=expand_dims.keys(),
                ),
            }
        )
        cls.reduce_dims = ("t", "l")

        # get scores (with components) from the inner score function
        cls.nse_score = nse_impl._nse_metascore(
            nse_impl.NseMetaInput(
                nse_impl.NseDatasets(
                    fcst=cls.ds_fcst,
                    obs=cls.ds_obs,
                    weights=None,
                ),
                gathered_dims=cls.reduce_dims,
                is_dataarray=True,
                is_angular=False,
            )
        )

        cls.nse_score_weights = nse_impl._nse_metascore(
            nse_impl.NseMetaInput(
                nse_impl.NseDatasets(
                    fcst=cls.ds_fcst,
                    obs=cls.ds_obs,
                    weights=cls.ds_weights,
                ),
                gathered_dims=cls.reduce_dims,
                is_dataarray=True,
                is_angular=False,
            )
        )

    common_params = [
        dict(var_="temperature", use_weights=False),
        dict(var_="precipitation", use_weights=False),
        dict(var_="temperature", use_weights=True),
        dict(var_="precipitation", use_weights=True),
    ]

    params = {
        "test_obs_variance": common_params,
        "test_fcst_error": common_params,
        "test_nse_score": common_params,
        "test_nse_against_naive_impl": common_params,
    }

    def test_obs_variance(self, var_, use_weights):
        """
        Tests obs variance is the same as the handcrafted scenario
        """
        res = self.nse_score.components.obs_variance
        exp = self.exp_obs_variance
        if use_weights:
            res = self.nse_score_weights.components.obs_variance
            exp = self.exp_obs_variance_weights
        xr.testing.assert_allclose(res.data_vars[var_], exp[var_])

    def test_fcst_error(self, var_, use_weights):
        """
        Tests forecast error is the same as the handcrafted scenario
        """
        res = self.nse_score.components.fcst_error
        exp = self.exp_fcst_error
        if use_weights:
            res = self.nse_score_weights.components.fcst_error
            exp = self.exp_fcst_error_weights
        xr.testing.assert_allclose(res.data_vars[var_], exp[var_])

    def test_nse_score(self, var_, use_weights):
        """
        Tests NSE score is the same as the handcrafted scenario
        """
        res = self.nse_score.components.nse
        exp = self.exp_nse_score
        if use_weights:
            res = self.nse_score_weights.components.nse
            exp = self.exp_nse_score_weights
        xr.testing.assert_allclose(res.data_vars[var_], exp[var_])

    def test_nse_against_naive_impl(self, var_, use_weights):
        """
        Tests against naive implementation of NSE (using for loops)
        """
        np_obs = self.ds_obs[var_].to_numpy()
        np_fcst = self.ds_fcst[var_].to_numpy()
        # no weights == all weights = 1.0
        np_weights = np.full_like(np_fcst, fill_value=1.0)

        # compute result
        res = self.nse_score.components.nse

        if use_weights:
            res = self.nse_score_weights.components.nse
            np_weights = self.ds_weights[var_].to_numpy()

        # compute nse_naive algorithm (loops)
        exp = NseSetup.nse_naive(np_fcst, np_obs, np_weights)

        assert np.allclose(res.data_vars[var_].to_numpy(), exp)


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
        res = nse_impl.nse(ds_fcst, ds_obs, reduce_dims=reduce_dims)
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

    @pytest.fixture(scope="class", autouse=True)
    def skip_if_dask_unavailable(self):
        """
        fixture to skip dask if it doesn't exist
        """
        if not DASK_AVAILABLE:
            pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

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
        tmp_da1_path = os.path.join(tmpdir, "da1.nc")
        tmp_da2_path = os.path.join(tmpdir, "da2.nc")
        da1.to_netcdf(tmp_da1_path)
        da2.to_netcdf(tmp_da2_path)

        # try NSE with dask
        with (
            xr.open_dataarray(tmp_da1_path, chunks=chunks) as da1_disk,
            xr.open_dataarray(tmp_da2_path, chunks=chunks) as da2_disk,
        ):
            res = nse_impl.nse(da1_disk, da2_disk, reduce_dims=("x", "y"))
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
