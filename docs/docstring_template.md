# Docstring Template
--------------------

```
def function_name(*args, *, **kwargs) -> type hint:
	'''
	A short one or two line description of what the function does. 

    Additional information (e.g. a paragraph or two) may be included here following the short description.

	MathJax giving the mathematical formula for the metric. Below is some MathJax to use as a 
	starting point. Note that double-backslash formatting is required to ensure proper rendering in docstrings.

    .. math::
        \\text{mean error} =\\frac{1}{N}\\sum_{i=1}^{N}(x_i - y_i)
        \\text{where } x = \\text{the forecast, and } y = \\text{the observation}	

	Where applicable, say what the range of the score is. For example:
	Range: 0 to infinity, 0 is perfect. 

	Mathjax can also be included inline, like this :math:`\\text{obs_cdf}(x) = 0`

	Args: 
        fcst: Forecast data. 
        obs: Observation data.
        reduce_dims: Dimensions to reduce. Can be "all" to reduce all dimensions.
        preserve_dims: Dimensions to preserve. Can be "all" to preserve all dimensions.
        weights: Weights for calculating a weighted mean of individual scores (e.g. by area, by latitude, 
            by population, custom). Note that these weights are different to threshold weighting which is done 
            by decision threshold.

    Returns:
    	A semantic description of what is returned. Note - type information is automatically included based on type hinting.

    Raises:
    	A type and description of any special error checking which may result in an exception, e.g.
        ValueError: if there are values in `fcst` and `obs` which are invalid

    Notes:
        Any additional comments or notes should go here.        

    References:
        - If possible, a citation to the first (original) paper introducing the score/metric.
        - In addition, if that paper is not open access, please also add an open access reference.
        - The preferred referencing style for journal articles is [APA (7th edition)](https://apastyle.apa.org/style-grammar-guidelines/references/examples/journal-article-references)
        - Example reference below:
        - Sorooshian, S., Duan, Q., & Gupta, V. K. (1993). Calibration of rainfall-runoff models:
          Application of global optimization to the Sacramento Soil Moisture Accounting Model.
          Water Resources Research, 29(4), 1185-1194. https://doi.org/10.1029/92WR02617        
        - Optionally, if there is a website(s) that has particularly good information about the metric, please feel free to list it as well.

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