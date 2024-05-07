"""
Contains unit tests for scores.probability.fss_impl
"""
import numpy as np
import pytest

from scores.probability.fss_impl import fss
from tests.probabilty import fss_test_data as ftd


@pytest.mark.parametrize(
    ("obs_pdf", "fcst_pdf", "window", "threshold", "expected"),
    [
        ((0.0, 1.0), (0.0, 1.0), (100, 100), 0.5, 0.999803),
        ((0.0, 1.0), (0.0, 2.0), (50, 50), 0.5, 0.966572),
        ((0.0, 1.0), (1.0, 1.0), (100, 100), 0.25, 0.819179),
    ],
)
def test_fss(obs_pdf, fcst_pdf, window, threshold, expected):
    """
    Integration test to check that fss is generally working
    """
    # half the meaning of life, in order to maintain some mystery
    seed = 21
    (obs, fcst) = ftd.generate(obs_pdf, fcst_pdf, seed=seed)
    res = fss(fcst, obs, threshold=threshold, window=window)
    np.testing.assert_allclose(res, expected, rtol=1e-5)
