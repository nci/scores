"""Tests the dask import checking utilties"""

import importlib
import sys
from unittest import mock

from scores import utils


def test_dask_fallback_block_execution():
    """
    Truly executes the 'else' block in scores/utils.py by reloading
    under a mock, then restores the module to its original state.
    In previous tests, hiding dask interfered with other tests.
    """
    # Capture the original classes to prevent "Version Mismatch"
    original_dimension_error = utils.DimensionError

    try:
        # 1. Hide dask
        with mock.patch.dict(sys.modules, {"dask": None}):
            # 2. Force the 'else' block to run
            importlib.reload(utils)

            # 3. Verify the 'else' block logic
            assert utils.HAS_DASK is False
            assert utils.da is None
            # This confirms the 'def is_dask_collection' in the else block was defined
            assert utils.is_dask_collection("anything") is False

    finally:
        # 4. RESTORE EVERYTHING
        # First, reload normally to bring Dask back
        importlib.reload(utils)

        # Second, manually put the original classes back into the module
        # This fixes the failures in test_utils.py because it ensures
        # utils.DimensionError is the exact same object it was at the start.
        utils.DimensionError = original_dimension_error

        # If you use other classes/functions in test_utils.py parametrization,
        # restore them here too:
        # utils.check_binary = original_check_binary


def test_verify_restoration():
    """Ensures Dask is back for subsequent tests."""
    assert utils.HAS_DASK is True
    assert utils.da is not None
