# Docstring Spec
-----------------

`scores` includes a lot of information in its docstrings. These docstrings are used in rendering the readthedocs API documentation, can be built locally as web pages using `sphinx`, can be seen in notebooks and the command-line using the Python's help functionality, or may be read as source code. The tech stack for all of this is somewhat complex. The docstrings include mathematical formulae, references, general links to further reading, coding information (like type information) as well as explanations for users.

This document attempts to set out the contents, layout and syntax which should be adopted where possible, including some information on common issues and gotchas. 

The authors are not specification experts, so this spec is not written in a formal spec language.

For convenience, this spec begins with a template.

# Docstring Template
--------------------

```
def function_name(*args, *, **kwargs) -> type hint:
	'''
	A short one or two line description of what the function does.

	Mathjax if at all possible giving the mathematical formula for the score. Here is some mathjax to use as a 
	starting point.

    .. math::
        \\text{mean error} =\\frac{1}{N}\\sum_{i=1}^{N}(x_i - y_i)
        \\text{where } x = \\text{the forecast, and } y = \\text{the observation}	

	Standard information on the value of the score. For example:
	Range: 0 to 1, 0 is perfect. 

	Mathjax can also be included inline, like this :math:`\\text{obs_cdf}(x) = 0`

	Args: 
        fcst: Forecast data. 
        obs: Observation data.
        reduce_dims: Dimensions to reduce. Can be "all" to reduce all dimensions.
        preserve_dims: Dimensions to preserve. Can be "all" to preserve all dimensions.
        weights: Weights for calculating a weighted mean of individual scores. Note that
            these weights are different to threshold weighting which is done by decision
            threshold.
    Returns:
    	A semantic description of what is returned. Type information is handled by type hinting already.
    Raises:
    	A description of any special error checking which may result in an exception
    References:
        If possible, a citation to the first (original) paper introducing the score.
        In addition, if that citation is not open access, add an open access link also.
        If there is general material like a wikipedia link, other docs site etc, add it here also
    See also:
    	If there are closely related functions, add e.g. :py:func:`scores.continuous.rmse` with a note
    Examples:

        >>> import numpy as np  # NOTE - the previous line should be empty for rendering to work properly.
        >>> import xarray as xr
        >>> from scores.probability import interval_tw_crps_for_ensemble
        >>> fcst = xr.DataArray(np.random.uniform(-40, 10, size=(10, 10)), dims=['time', 'ensemble'])
        >>> obs = xr.DataArray(np.random.uniform(-40, 10, size=10), dims=['time'])


	'''
```