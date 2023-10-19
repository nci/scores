"""
`scores` `torch` wrapper
"""


from typing import Any, Union
import torch
import xarray as xr

from scores.experimental.wrapper import APIWrapper

class PyTorch(APIWrapper):
    """
    Specific wrapper to allow use of `scores` with `torch`.
    """
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, dataset_select: Union[str, None] = None, **kwargs) -> torch.Tensor:
        """
        Call underlying function, handling `torch.Tensor`

        Due to `torch` implementation only operations on Tensor can be used.

        Args:
            prediction (torch.Tensor): 
                Predicted Tensor
            target (torch.Tensor): 
                Target Tensor
            dataset_select (Union[str, None], optional): 
                Variable to select, if score is an `xarray.Dataset`. Defaults to None.

        Raises:
            ValueError: 
                If score was a xr.Dataset and `dataset_select` not given.
            ValueError: 
                If `dataset_select` given, and score not a xr.Dataset

        Returns:
            (torch.Tensor): 
                score as calculated
        """        

        ## Note: Cannot detach tensors
        ##       will not work with scores using xarray

        # if hasattr(prediction, 'detach'):
        #     prediction = prediction.detach()
        #     target = target.detach()

        # prediction = xr.DataArray(prediction.numpy())
        # target = xr.DataArray(target.numpy())
                
        score = super().__call__(prediction, target, **kwargs)

        if isinstance(score, xr.Dataset):
            if dataset_select is None:
                raise ValueError(
                    f"Returned score is an 'xarray.Dataset' which cannot be parsed to a Tensor.\n"
                    "Set, `dataset_select` to a variable to select that variable."
                    )
            score = score[dataset_select]

        elif dataset_select:
            raise ValueError(f"`dataset_select` was not None, but data was not a 'xarray.Dataset'.")

        if isinstance(score, xr.DataArray):
            score = score.values
        return torch.Tensor(score)
    