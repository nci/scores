# disable non-critical pylinting for tests:
# pylint: disable=use-dict-literal
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-few-public-methods
"""
Collection of tests for the NSE score, contains:
    - NseSetup: Base class used by all other test classes
    - TestNsePublicApi: Tests concerning the public API interface - this is the main suite
    - TestNseScoreBuilder: Tests specific to the builder that are not covered by public API
    - TestNseUtils: Tests specific to the utils that are not covered by public API
    - TestNseScore: Tests specific to score computation not already covered by public API
    - TestNseDataset: Tests compatiblity with datasets (most tests use data array for convenience)
    - TestNseDask: Tests compatibility with dask
"""
import os

import numpy as np
import pytest
import xarray as xr
from numpy import typing as npt

from scores.continuous import nse, nse_impl
from scores.utils import DimensionError

DASK_AVAILABLE = False
try:
    import dask
    import dask.array

    DASK_AVAILABLE = True
except ImportError:
    pass


class NseSetup:
    """
    Base class for NSE tests with some setup and helper functions
    """

    _SEED: int = 42

    @staticmethod
    def make_random_xr_array(
        shape: tuple[int],
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

        # used mainly by NseScore as a helper
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


    NOTE: this implicitly tests most of the ``NseScoreBuilder`` class, since it essentially
    dispatches any checks to that class, as such they are not repeated as long as coverage is happy.
    """

    @classmethod
    def setup_class(cls):
        """
        Common data to be reused by the rest of this class
        """
        # --- input arrays ---
        # make the default obs/fcst have different seeds so we don't get back the same array.
        # default obs
        cls.obs = NseSetup.make_random_xr_array(
            shape=(4, 2, 3),
            dim_names=["t", "x", "y"],
            override_seed=42,
        )
        # default fcst
        cls.fcst = NseSetup.make_random_xr_array(
            shape=(4, 2, 3),
            dim_names=["t", "x", "y"],
            override_seed=24,
        )
        # bad obs name
        # note: all the names must be wrong, if some names are correct
        cls.obs_badnames = NseSetup.make_random_xr_array(
            shape=(4, 2, 3),
            dim_names=["t_bad", "x_bad", "y_bad"],
        )
        # incorrect dim sizes between arrays (non broadcastable)
        cls.obs_baddimsizes = NseSetup.make_random_xr_array(
            shape=(5, 1, 2),
            dim_names=["t", "x", "y"],
        )

        # --- dimension specification ---
        # good dimension specifications
        cls.reduce_dim_string = "t"  # should give (x, y) array
        cls.reduce_dims_list = ["t", "x"]  # should give (y, 1) array
        cls.reduce_dim_all = "all"  # single value output
        cls.preserve_dim_string = "y"  # should give (y, 1) array
        cls.preserve_dim_list = ["x", "y"]  # should give (x, y) array
        # bad dimension specifications
        cls.preserve_dim_all1 = "all"  # NSE cannot be computed, no dimensions reduced
        cls.preserve_dim_all2 = ["x", "y", "t"]  # ditto above
        cls.reduce_dims_none = []  # ditto above

        # --- insufficient data to reduce ---
        cls.obs_insufficient_data = NseSetup.make_random_xr_array(
            shape=(4, 1, 1),
            dim_names=["t", "x", "y"],
            override_seed=42,
        )
        cls.fcst_insufficient_data = NseSetup.make_random_xr_array(
            shape=(4, 1, 1),
            dim_names=["t", "x", "y"],
            override_seed=24,
        )
        cls.reduce_dims_insufficient_data = ["x", "y"]

        # --- weights ---
        # default weights
        #   NOTE: ideally weights should be specified explicitly for each index, but broadcasting is
        #   currently allowed (and potentially okay, as long as the dimension shapes conform for
        #   the specified axes.)
        cls.weights = NseSetup.make_random_xr_array(
            shape=(4, 2),
            dim_names=["t", "x"],
        )
        # these should still work, as "nan" acts as exclusion and 0 acts as zero-forcing
        cls.weights.loc[dict(x=1, t=0)] = 0.0  # make one of the weights 0
        cls.weights.loc[dict(t=1, x=0)] = np.nan  # make one of the weights np.nan
        # failure conditions for weights
        cls.negative_weights = cls.weights.copy(deep=True)
        cls.negative_weights.loc[dict(t=0, x=0)] = -1.0  # make one of the weights negative
        cls.allzero_weights = cls.weights.copy(
            deep=True,
            data=np.zeros(cls.weights.shape),
        )
        cls.allnan_weights = cls.weights.copy(
            deep=True,
            data=np.full(cls.weights.shape, fill_value=np.nan),
        )

        # --- divide by zero ---
        # set the value along one index of the dimension being reduced: "x,y" , as all one
        # this will force a divide by zero error in one entry in the resulting array, which should
        # still allow the calculation to be completed.
        cls.reduce_dims_divide_by_zero = ["x", "y"]
        # obs_divide_by_zero v.s. fcst (default) = -np.inf only on the plane where this happens
        cls.obs_divide_by_zero = cls.obs.copy(deep=True)
        cls.obs_divide_by_zero.loc[dict(t=1)] = 42.123
        # obs_divide_by_zero v.s. fcst_divide_by_zero = np.nan, ditto above
        cls.fcst_divide_by_zero = cls.fcst.copy(deep=True)
        cls.fcst_divide_by_zero.loc[dict(t=1)] = 42.123

    def test_error_incompatible_fcst_obs_dimnames(self):
        """
        If there are no common dimension names we should expect a ValueError
        """
        with pytest.raises(ValueError):
            nse(self.fcst, self.obs_badnames)

    def test_error_incompatible_dimsizes(self):
        """
        If the dimension sizes don't match we should expect a ValueError
        """
        with pytest.raises(ValueError):
            nse(self.fcst, self.obs_baddimsizes, reduce_dims=self.reduce_dim_string)

    def test_error_invalid_dims_specification(self):
        """
        A battery of tests to check for invalid input specifications for preserve_dims and
        reduce_dims.
        """
        # preserve=all
        with pytest.raises(DimensionError):
            nse(self.fcst, self.obs, preserve_dims=self.preserve_dim_all1)
        # essentially the same as above but explicitly specified - see class setup
        with pytest.raises(DimensionError):
            nse(self.fcst, self.obs, preserve_dims=self.preserve_dim_all2)
        # reduce_dims is empty - also essentially the same
        with pytest.raises(DimensionError):
            nse(self.fcst, self.obs, reduce_dims=self.reduce_dims_none)
        # both options specified (safety check) - note: overspecified args (i.e. both reduce_dims and preserve_dims) is a ValueError
        with pytest.raises(ValueError):
            nse(
                self.fcst,
                self.obs,
                reduce_dims=self.reduce_dim_string,
                preserve_dims=self.preserve_dim_list,
            )

    def test_error_reduced_dims_not_enough_data_for_obs_variance(self):
        """
        Should raise DimensionError if the theres only one item to be reduced, as this cannot be
        used to compute the observation variance, making the score useless.
        """
        with pytest.raises(DimensionError):
            nse(
                self.fcst_insufficient_data,
                self.obs_insufficient_data,
                reduce_dims=self.reduce_dims_insufficient_data,
            )

    def test_error_invalid_weights(self):
        """
        Should raise an error if weights:
            - contain a negative element, and the following cases raise errors in case of
              unintentional inputs.
            - are all nans (everything is masked - nothing to compute)
            - are all zeros (everything is zero forced - score is NaN)
        """
        with pytest.raises(ValueError):
            nse(
                self.fcst,
                self.obs,
                weights=self.negative_weights,
                reduce_dims=self.reduce_dims_list,
            )
        with pytest.raises(ValueError):
            nse(
                self.fcst,
                self.obs,
                weights=self.allnan_weights,
                reduce_dims=self.reduce_dims_list,
            )
        with pytest.raises(ValueError):
            nse(
                self.fcst,
                self.obs,
                weights=self.allzero_weights,
                reduce_dims=self.reduce_dims_list,
            )

    def test_warn_divide_by_zero(self):
        """
        Should warn when divide by zero error happens, but not raise an error, tests two cases:
            - when both obs and fcst are 0 - should have a NaN result
            - when only obs is 0 - should have -Inf in the result
        """
        # should have one fabricated -Inf entry at t=1
        with pytest.warns(UserWarning):
            ret = nse(
                self.fcst,
                self.obs_divide_by_zero,
                reduce_dims=self.reduce_dims_divide_by_zero,
            )
            assert np.any(np.isneginf(ret[1]))
        # should have one fabricated NaN entry at t=1
        with pytest.warns(UserWarning):
            ret = nse(
                self.fcst_divide_by_zero,
                self.obs_divide_by_zero,
                reduce_dims=self.reduce_dims_divide_by_zero,
            )
            assert np.any(np.isnan(ret[1]))

    def test_minimal_default_behaviour(self):
        """
        Tests minimal NSE score function
        """
        # reduce_dims and preserve_dims = None => everything reduced
        ret = nse(self.fcst, self.obs)
        assert np.all(ret <= 1.0)

    def test_default_behaviour_with_kitchen_sink(self):
        """
        Tests NSE score function with all args/kwargs
        """
        # using obs as weights is quite common
        ret = nse(
            self.fcst,
            self.obs,
            weights=np.abs(self.obs),
            reduce_dims=None,
            preserve_dims=["x", "y"],
            is_angular=False,
        )
        # expect x by y array returned
        assert "x" in ret.dims
        assert "y" in ret.dims
        assert ret.shape == (2, 3)
        assert np.all(ret <= 1.0)

    def test_default_with_angular_data(self):
        """
        Tests NSE score function with angular data
        """
        ret = nse(self.fcst * 360, self.obs * 360, reduce_dims="t", is_angular=True)
        assert np.all(ret <= 1.0)


class TestNseScoreBuilder:
    """
    NOTE: most of NseScoreBuilder is tested by the public API test suite and is not repeated here.
    Only things missed by the public API test suite will be covered here.
    """

    def test_invalid_builder_initialization(self):
        """
        Tests if runtime error is raised if builder is used more than once in the same scope.
        This is bad, as it will cause different score functions to share the same reference data.
        """
        obs = NseSetup.make_random_xr_array((1, 2), ["x", "y"])
        fcst = NseSetup.make_random_xr_array((1, 2), ["x", "y"])

        with pytest.raises(RuntimeError):
            bld = nse_impl.NseScoreBuilder()
            # ok
            # intentionally disable unused-variable to test for double builder usage error
            score1 = bld.build(fcst=fcst, obs=obs)  # pylint: disable=unused-variable
            score2 = bld.build(fcst=fcst, obs=obs)  # pylint: disable=unused-variable


class TestNseUtils(NseSetup):
    """
    NOTE: most of NseScoreBuilder is tested by the public API test suite and is not repeated here.
    Only things missed by the public API test suite will be covered here.
    """

    def test_get_xr_type_marker_mixed_type_error(self):
        """
        Tests error is raised when mixed input types are given.
        """
        da = NseSetup.make_random_xr_array((1, 2), ["x", "y"])
        ds = xr.Dataset(dict(mix_and_match=da))
        dw = None

        with pytest.raises(TypeError):
            nse_impl.NseUtils.get_xr_type_marker(da, ds, dw)


class TestNseScore(NseSetup):
    """
    Tests validity of the actual NSE scoring functions for a given set of inputs. As such, tests
    here should not use ``np.random.random``, instead they'd need to be either handcrafted or
    verifiable by a naive secondary algorithm.
    """

    @classmethod
    def setup_class(cls):
        """
        TODO: move this to setup class, and create one test per expected output

        Worked example - note this is not a scientific example, its still very much contrived, but
        still verifies some important properties of NSE.  Please have a look at the tutorial
        notebooks for a simulated hydrograph example.

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

        # ---------------------------------------
        #  scores - prepped but not yet computed
        # ---------------------------------------
        # expose score values by assigning them to the class
        # these (lazy) computations are used for most tests
        cls.nse_score = nse_impl.NseScoreBuilder().build(
            fcst=cls.ds_fcst,
            obs=cls.ds_obs,
            reduce_dims=cls.reduce_dims,
        )

        cls.nse_score_weights = nse_impl.NseScoreBuilder().build(
            fcst=cls.ds_fcst,
            obs=cls.ds_obs,
            weights=cls.ds_weights,
            reduce_dims=cls.reduce_dims,
        )

    def test_invalid_score_initialization(self):
        """
        Tests scenario where score is initialized without a builder. This is bad, as a builder
        performs all the checks from interfacing with the public API.
        """
        obs = NseSetup.make_random_xr_array((1, 2), ["x", "y"])
        fcst = NseSetup.make_random_xr_array((1, 2), ["x", "y"])
        weights = NseSetup.make_random_xr_array((1, 2), ["x", "y"])

        with pytest.raises(RuntimeError):
            nse_impl.NseScore(
                fcst=fcst,
                obs=obs,
                weights=weights,
                xr_type_marker=1,
                reduce_dims="x",
                ref_builder=None,
            )

    @pytest.mark.parametrize(
        "var_,use_weights",
        [
            ("temperature", False),
            ("precipitation", False),
            ("temperature", True),
            ("precipitation", True),
        ],
    )
    def test_obs_variance(self, var_, use_weights):
        """
        Tests obs variance is the same as the handcrafted scenario
        """
        res = self.nse_score.obs_variance
        exp = self.exp_obs_variance
        if use_weights:
            res = self.nse_score_weights.obs_variance
            exp = self.exp_obs_variance_weights
        xr.testing.assert_allclose(res.ds.data_vars[var_], exp[var_])

    @pytest.mark.parametrize(
        "var_,use_weights",
        [
            ("temperature", False),
            ("precipitation", False),
            ("temperature", True),
            ("precipitation", True),
        ],
    )
    def test_fcst_error(self, var_, use_weights):
        """
        Tests forecast error is the same as the handcrafted scenario
        """
        res = self.nse_score.fcst_error
        exp = self.exp_fcst_error
        if use_weights:
            res = self.nse_score_weights.fcst_error
            exp = self.exp_fcst_error_weights
        xr.testing.assert_allclose(res.ds.data_vars[var_], exp[var_])

    @pytest.mark.parametrize(
        "var_,use_weights",
        [
            ("temperature", False),
            ("precipitation", False),
            ("temperature", True),
            ("precipitation", True),
        ],
    )
    def test_nse_score(self, var_, use_weights):
        """
        Tests NSE score is the same as the handcrafted scenario
        """
        res = self.nse_score.nse
        exp = self.exp_nse_score
        if use_weights:
            res = self.nse_score_weights.nse
            exp = self.exp_nse_score_weights
        xr.testing.assert_allclose(res.data_vars[var_], exp[var_])

    @pytest.mark.parametrize(
        "var_,use_weights",
        [
            ("temperature", False),
            ("precipitation", False),
            ("temperature", True),
            ("precipitation", True),
        ],
    )
    def test_nse_against_naive_impl(self, var_, use_weights):
        np_obs = self.ds_obs[var_].to_numpy()
        np_fcst = self.ds_fcst[var_].to_numpy()
        # no weights == all weights = 1.0
        np_weights = np.full_like(np_fcst, fill_value=1.0)

        # compute result
        res = self.nse_score.nse

        if use_weights:
            res = self.nse_score_weights.nse
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
        res = nse(ds_fcst, ds_obs, reduce_dims=reduce_dims)
        # result is a dataset
        assert isinstance(res, xr.Dataset)
        # variables are data arrays
        assert isinstance(res["precip"], xr.DataArray)
        assert isinstance(res["temp"], xr.DataArray)
        # precip should only have "t", since "h" isn't defined for either obs or fcst in precip
        assert set(res["precip"].dims) == set(["t"])
        assert res["precip"].shape == (2,)
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
            res = nse(da1_disk, da2_disk, reduce_dims=("x", "y"))
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
