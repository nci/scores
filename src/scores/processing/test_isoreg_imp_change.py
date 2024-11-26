"""Module to test the change from the scikit-learn implementation of isotonic regression
to a scipy implementation

Passes the same x, y, and weight arrays to two functions 
(one scikit-learn and one scipy)
and asserts the resulting arrays are the same.

Runs the test an arbitrary 50 times
"""

from functools import partial
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from numpy.testing import assert_array_equal
from scipy.optimize import isotonic_regression
from sklearn.isotonic import IsotonicRegression


def test_compare_funcs():
    """Run a test to compare the outputs of the scikit learn based isotonic function
    and the new scipy based implementation
    """
    # Arbitrary 50 runs
    for i in range(50):
        # Create the test data in each loop for x,y, and weights
        x_arr = np.random.randint(1, 10, 10)
        y_arr = np.random.randint(1, 10, 10)
        weight = np.random.randint(1, 10, 10)

        # IMPORTANT:
        # y array is sorted descending,
        # x array is sortined ascending
        # per _tidy_ir_inputs function in the original module
        x_arr = np.sort(x_arr)
        y_arr = np.sort(y_arr)[::-1]

        # Confirm the inputs
        print("x_arr ", x_arr)
        print("y_arr ", y_arr)
        print("weight ", weight)

        # Calculate ouputs
        scikit_result_array = _deprecated_contiguous_mean_ir(x=x_arr, y=y_arr, weight=weight)
        scipy_result_array = _contiguous_mean_ir(x=x_arr, y=y_arr, weight=weight)

        # Confirm outputs from the two calculation methods
        print("Scikit: ", scikit_result_array)
        print("Scipy: ", scipy_result_array)

        # Main test: assert the same arrays
        assert_array_equal(scikit_result_array, scipy_result_array)

    print("All tests passed!")


def _contiguous_mean_ir(
    x: np.ndarray, y: np.ndarray, *, weight: Optional[np.ndarray] = None  # Force keywords arguments to be keyword-only
) -> np.ndarray:
    """
    Performs classical (i.e. for mean functional) contiguous quantile IR on tidied data x, y.
    Refactored to use scipy instead of scikit learn.

    IMPORTANT NOTE: the y array MUST be sorted in descending order before being passed to the function. This
    implicitly handled by the _tidy_ir_inputs function before calling this function.
    To make explicit could include y = np.sort(y)[::-1]
    """
    return isotonic_regression(y, weights=weight, increasing=True).x  # type: ignore


def _deprecated_contiguous_mean_ir(
    x: np.ndarray, y: np.ndarray, *, weight: Optional[np.ndarray] = None  # Force keywords arguments to be keyword-only
) -> np.ndarray:
    """
    Performs classical (i.e. for mean functional) contiguous quantile IR on tidied data x, y.
    Uses sklearn implementation rather than supplying the mean solver function to `_contiguous_ir`,
    as it is about 4 times faster (since it is optimised for mean).
    """
    return IsotonicRegression().fit_transform(x, y, sample_weight=weight)  # type: ignore


if __name__ == "__main__":
    test_compare_funcs()
