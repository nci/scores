"""Tests the dask import checking utilties"""

import importlib
import sys
from unittest import mock

from scores import utils


def test_dask_fallback_block_execution():
    """
    Truly executes the 'else' block in scores/utils.py and restores
    the module to its pre-test state, regardless of the CI environment.
    """
    # 1. Capture the state of the module before messing with it
    # This includes HAS_DASK (True or False), da, and the classes
    original_state = {
        "HAS_DASK": utils.HAS_DASK,
        "dask": utils.dask,
        "is_dask_collection": utils.is_dask_collection,
        "DimensionError": utils.DimensionError,
    }

    try:
        # 2. Force the 'else' block to run by hiding dask
        with mock.patch.dict(sys.modules, {"dask": None}):
            importlib.reload(utils)

            # 3. Verify the fallback logic executed
            assert utils.HAS_DASK is False
            assert utils.dask is None
            assert utils.is_dask_collection("anything") is False

    finally:
        # 4. Restore the module to exactly how we found it
        # This prevents breaking the pipeline if Dask was missing,
        # and prevents breaking test_utils.py if Dask was present.
        for key, value in original_state.items():
            setattr(utils, key, value)


def test_verify_restoration():
    """
    Ensures the module returned to its original state.
    Uses find_spec to check for dask availability without unused imports.
    """
    # check if dask is actually installed in the environment
    dask_installed = importlib.util.find_spec("dask") is not None

    # utils.HAS_DASK should match the actual environment state
    assert utils.HAS_DASK == dask_installed
