
# API Documentation

```{contents} Table of Contents
   :depth: 1
   :local:
   :backlinks: none
```


## scores.continuous
```{eval-rst}
.. automodule:: scores.continuous
.. autofunction:: scores.continuous.additive_bias
.. autofunction:: scores.continuous.mean_error
.. autofunction:: scores.continuous.mae
.. autofunction:: scores.continuous.mse
.. autofunction:: scores.continuous.quantile_score
.. autofunction:: scores.continuous.rmse
.. autofunction:: scores.continuous.murphy_score
.. autofunction:: scores.continuous.murphy_thetas
.. autofunction:: scores.continuous.flip_flop_index
.. autofunction:: scores.continuous.flip_flop_index_proportion_exceeding
.. autofunction:: scores.continuous.correlation
.. autofunction:: scores.continuous.isotonic_fit
```

## scores.probability
```{eval-rst}
.. autofunction:: scores.probability.crps_cdf
.. autofunction:: scores.probability.adjust_fcst_for_crps
.. autofunction:: scores.probability.crps_cdf_brier_decomposition
.. autofunction:: scores.probability.crps_for_ensemble
.. autofunction:: scores.probability.murphy_score
.. autofunction:: scores.probability.murphy_thetas
.. autofunction:: scores.probability.roc_curve_data
.. autofunction:: scores.probability.brier_score
.. autofunction:: scores.probability.isotonic_fit
```

## scores.categorical
```{eval-rst}
.. autofunction:: scores.categorical.firm
.. autofunction:: scores.categorical.probability_of_detection
.. autofunction:: scores.categorical.probability_of_false_detection
```

## scores.processing
```{eval-rst}
.. autofunction:: scores.processing.isotonic_fit
.. autofunction:: scores.processing.broadcast_and_match_nan
.. autofunction:: scores.processing.comparative_discretise
.. autofunction:: scores.processing.binary_discretise
.. autofunction:: scores.processing.binary_discretise_proportion
.. autofunction:: scores.processing.proportion_exceeding
.. autofunction:: scores.processing.cdf.round_values
.. autofunction:: scores.processing.cdf.propagate_nan
.. autofunction:: scores.processing.cdf.observed_cdf
.. autofunction:: scores.processing.cdf.integrate_square_piecewise_linear
.. autofunction:: scores.processing.cdf.add_thresholds
.. autofunction:: scores.processing.cdf.fill_cdf
.. autofunction:: scores.processing.cdf.decreasing_cdfs
.. autofunction:: scores.processing.cdf.cdf_envelope
```

## scores.stats
```{eval-rst}
.. autofunction:: scores.stats.statistical_tests.diebold_mariano
```

## scores.pandas
```{eval-rst}
.. autofunction:: scores.pandas.continuous.mse
.. autofunction:: scores.pandas.continuous.rmse
.. autofunction:: scores.pandas.continuous.mae
```