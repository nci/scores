"""
Context managers for `scores`
"""
from scores import experimental

class APIChange:
    def __init__(self, api_name: str):
        """Change the api of `scores` as defined

        Args:
            api_name (str): 
                Name of `api` to change to

        Example:
            >>> import scores.experimental
            >>> with scores.experimental.source('NEWAPI'):
                    scores.experimental.api.continuous # Will be wrapper around said function
        """
        if not hasattr(experimental, api_name):
            raise AttributeError(f"`scores.experimental` has no attribute {api_name}.")
        self.api_name = api_name
        self._recorded_api = None

    def __enter__(self):
        """Change api"""
        self._recorded_api = experimental.api
        setattr(experimental, 'api', getattr(experimental, self.api_name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(experimental, 'api', self._recorded_api)
