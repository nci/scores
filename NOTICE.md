This file contains a listing of all code which is included within the repository which is adapted from, extends, or includes code from another open source project. The filenames, sources, copyright and licenses are recorded here.

Code in the file src/scores/probability/crps_numba.py is adapted from two sources. 
The xarray wrapper function crps_cdf_exact_fast is based on the code for crps_ensemble from xskillscore
https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/probabilistic.py, released under the  Apache-2.0 License with 
copyright attributed to xskillscore developers (2018-2021). The vectorisation of crps_at_point follows the example of _crps_ensemble_gufunc from properscoring
https://github.com/properscoring/properscoring/blob/master/properscoring/_gufuncs.py, released under the Apache-2.0 License with copyright attributed to The Climate Corporation (2015).