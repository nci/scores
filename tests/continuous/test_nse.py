# pylint: disable=use-dict-literal
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
import numpy as np
import pytest
import xarray as xr
from numpy import typing as npt

from scores.continuous import nse, nse_impl
from scores.utils import DimensionError


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
        # reduce_dims and preserve_dims = None => everything reduced
        ret = nse(self.fcst, self.obs)
        assert np.all(ret <= 1.0) and np.all(ret >= -np.inf)

    def test_default_behaviour_with_kitchen_sink(self):
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
        assert np.all(ret <= 1.0) and np.all(ret >= -np.inf)

    def test_default_with_angular_data(self):
        ret = nse(self.fcst * 360, self.obs * 360, reduce_dims="t", is_angular=True)
        assert np.all(ret <= 1.0) and np.all(ret >= -np.inf)


class TestNseScoreBuilder:
    """
    NOTE: most of NseScoreBuilder is tested by the public API test suite and is not repeated here.
    Only things missed by the public API test suite will be covered here.
    """

    def test_invalid_builder_initialization(self):
        obs = NseSetup.make_random_xr_array((1, 2), ["x", "y"])
        fcst = NseSetup.make_random_xr_array((1, 2), ["x", "y"])

        with pytest.raises(RuntimeError):
            bld = nse_impl.NseScoreBuilder()
            # ok
            score1 = bld.build(fcst=fcst, obs=obs)
            # double building not allowed before score1 is destructed.
            # Otherwise, score2 may implicitly be able to reference the same data.
            score2 = bld.build(fcst=fcst, obs=obs)


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

    def test_invalid_score_initialization(self):
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

    def test_nse_score_with_worked_example(self):
        """
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
                - x = 1: weights along l = [0,6,2]
                - since there is a slight assymetry in the error, we'd expect x=1 to have a slightly
                  smaller score - [smaller score]

            for x = 1:
                - fcst error =             8/3 = 2.6666....    (scaled by 2)
                - obs error  = (36+2+36+50)/24 = 63/12 = 5.25  (scaled by less than 2)

            +----------------------------------------------------------------+
            | NSE with weights = 1 - 8*12/3*63 = 31/63 = 0.492... ---> [4b]  |
            +----------------------------------------------------------------+

            - [smaller score]: as expected

            So the result will be for x=0: 0.6 and x=1: 0.492...

        ------------------------------------------------------------------------------------------
         Let's add another variable - it is a dataset after all
        ------------------------------------------------------------------------------------------
            Let's assume the above was for temperature, let's make our life harder and add another
            variable - precipitation. This time let's make the difference cycle between 2 and 0
            instead of 1 the average error is still 1 - but let's see if it affects the score

            i.e. fcst = [obs+2, obs+0, obs+2, obs+0, ...]

            +-------------------------------------------------------------------------+
            | now:                                                                    |
            |     - forecast error = (2^2 * 3) / 6 = 3: shock! error has increased!   |
            |     - obs variance = still the same (thankfully)                        |
            |     - nse without weights = 1 - (3 * 24 / 70)                           |
            |                           =  ~=-0.0285 (we're slightly worse than mean) |
            |     --> [6a,6b,6c]                                                      |
            +-------------------------------------------------------------------------+

            We're slighly worse off than using the mean, this also makes sense! We have effectively
            made our predictions less consistent/reliable and as a result more noisy even when
            compared to the obs. In fact the spread is 3 (fcst error) v.s. 2.91 (obs variance), as
            opposed to 1 v.s. 2.91 previously.  Eventhough, our _average_ error has not changed.
            Cool!

            lets add some cheeky weights now to conveniently only select the forecasts with 0
            errors. This time the result should be easy to work out i.e. if we have no error the
            best score NSE can give us is 1.0.

            How do we choose the weights? We use the selection pattern i.e. 0 if even, 1 if odd,
            assuming assuming the odd fcst indices match the obs and the evens are 2 away from the
            obs (.. or alternatively we could mask it - it doesn't matter for this particular
            score). [0, 1, 0, 1...]

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
        # -------------------------
        # x=2,y=4,t=2,l=3: total=48
        # -------------------------

        # build temperature dataarray - offset by 1:
        temp_obs = np.linspace(start=0, stop=48, dtype=int, endpoint=False)
        temp_fct = np.linspace(start=1, stop=49, dtype=int, endpoint=False)
        temp_obs = np.reshape(temp_obs, (2, 4, 2, 3))
        temp_fct = np.reshape(temp_fct, (2, 4, 2, 3))

        # build precipitation dataarray - offset by 2 on even entries:
        # actual start value shouldn't actually matter since the error is relative
        precip_obs = np.linspace(start=10, stop=58, dtype=int, endpoint=False)
        precip_fct = np.linspace(start=10, stop=58, dtype=int, endpoint=False)
        precip_fct[slice(0, None, 2)] += 2  # offset every even index
        precip_obs = np.reshape(precip_obs, (2, 4, 2, 3))
        precip_fct = np.reshape(precip_fct, (2, 4, 2, 3))

        # weights
        temp_weights = np.array([[[0, 3, 1]], [[0, 6, 2]]])  # x,.,.,l: 2*1*1*3
        precip_weights = np.array([[[0, 1, 0], [1, 0, 1]]])  # .,.,t,l: 1*1*2*3

        # expected values (needs to be broadcast appropriately)
        # temperature - scalar values
        exp_temp_fct_err = 1.0
        exp_temp_obs_var = 70.0 / 24.0
        exp_temp_nse_scr = 46.0 / 70.0
        exp_temp_fct_err_weights_a = 4.0 / 3.0
        exp_temp_obs_var_weights_a = 10.0 / 3.0
        exp_temp_nse_scr_weights_a = 3.0 / 5.0
        exp_temp_fct_err_weights_b = 8.0 / 3.0
        exp_temp_obs_var_weights_b = 63.0 / 12.0
        exp_temp_nse_scr_weights_b = 31.0 / 63.0
        # temperature - as broadcast array
        # TODO:

        # precipitation - scalar values
        exp_precip_fct_err = 3.0
        exp_precip_obs_var = 70.0 / 24.0
        exp_precip_nse_scr = -1.0 / 35.0
        exp_precip_fct_err_weights_a = 0.0
        exp_precip_obs_var_weights_a = 999  # irrelvant
        exp_precip_nse_scr_weights_a = 1.0
        exp_precip_fct_err_weights_b = 0.0
        exp_precip_obs_var_weights_b = 999  # irrelevant
        exp_precip_nse_scr_weights_b = 1.0

        # temperature - as broadcast array
        # TODO:

        # TODO: finish this test
        # - assert individual components
        # - assert final score
        # - assert compare against naive_nse


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
        assert not ("tapioca" in res.variables.keys())


class TestNseDask(NseSetup):
    """
    Basic testing if dask is available and used appropriately by NSE.

    NOTE: failure conditions will not be the responsibility of this test, this suite just exists to
    check if dask computes things appropriately with non-dask as a compatiblity measure.
    """

    # TODO: finish this test

    @pytest.fixture(scope="class", autouse=True)
    def skip_if_dask_unavailable(self):
        try:
            import dask
            import dask.array
        except ImportError:
            pytest.skip("Dask unavailable, could not run test")  # pragma: no cover
