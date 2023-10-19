"""
Experimental features in `scores`

All api's in here are subject to change, and may be moved into the main `scores` namespace.

Can use .api to access scores, which then the context manager `source` can change.
"""

from scores.experimental.wrapper import APIWrapper
from scores.experimental.context import APIChange as source

# Acts as standard api for scores
api = APIWrapper()

try:
    from scores.experimental.pytorch import PyTorch
    # Pytorch api
    pytorch = PyTorch()
except (ImportError, ModuleNotFoundError):
    pass
