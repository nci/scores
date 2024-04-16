
# API Documentation

```{contents} Table of Contents
   :depth: 1
   :local:
   :backlinks: none
```


## scores.continuous
```{eval-rst}
.. automodule:: scores.continuous
.. autofunction:: scores.continuous.mae
.. autofunction:: scores.continuous.mse
.. autofunction:: scores.probability.quantile_score
.. autofunction:: scores.continuous.rmse
.. autofunction:: scores.continuous.murphy_score
.. autofunction:: scores.continuous.murphy.thetas
.. autofunction:: scores.continuous.flip_flop_index
.. autofunction:: scores.continuous.flip_flop_index_proportion_exceeding
.. autofunction:: scores.continuous.isotonic_fit
.. autofunction:: scores.continuous.correlation
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
```

## scores.categorical
```{eval-rst}
.. autofunction:: scores.categorical.firm
.. autofunction:: scores.categorical.probability_of_detection
.. autofunction:: scores.categorical.probability_of_false_detection
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