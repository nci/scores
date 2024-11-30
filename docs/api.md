
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
.. autofunction:: scores.continuous.correlation.pearsonr
.. autofunction:: scores.continuous.multiplicative_bias
.. autofunction:: scores.continuous.pbias
.. autofunction:: scores.continuous.kge
.. autofunction:: scores.continuous.isotonic_fit
.. autofunction:: scores.continuous.consistent_expectile_score
.. autofunction:: scores.continuous.consistent_quantile_score
.. autofunction:: scores.continuous.consistent_huber_score
.. autofunction:: scores.continuous.tw_quantile_score
.. autofunction:: scores.continuous.tw_absolute_error
.. autofunction:: scores.continuous.tw_squared_error
.. autofunction:: scores.continuous.tw_huber_loss
.. autofunction:: scores.continuous.tw_expectile_score
.. autofunction:: scores.continuous.quantile_interval_score
.. autofunction:: scores.continuous.interval_score
```

## scores.probability
```{eval-rst}
.. autofunction:: scores.probability.crps_cdf
.. autofunction:: scores.probability.adjust_fcst_for_crps
.. autofunction:: scores.probability.crps_step_threshold_weight
.. autofunction:: scores.probability.crps_cdf_brier_decomposition
.. autofunction:: scores.probability.crps_for_ensemble
.. autofunction:: scores.probability.tw_crps_for_ensemble
.. autofunction:: scores.probability.tail_tw_crps_for_ensemble
.. autofunction:: scores.probability.interval_tw_crps_for_ensemble
.. autofunction:: scores.probability.murphy_score
.. autofunction:: scores.probability.murphy_thetas
.. autofunction:: scores.probability.roc_curve_data
.. autofunction:: scores.probability.brier_score
.. autofunction:: scores.probability.brier_score_for_ensemble
.. autofunction:: scores.probability.isotonic_fit
```

## scores.categorical
```{eval-rst}
.. autofunction:: scores.categorical.firm
.. autofunction:: scores.categorical.probability_of_detection
.. autofunction:: scores.categorical.probability_of_false_detection
.. autoclass:: scores.categorical.BinaryContingencyManager
    :members:
.. autoclass:: scores.categorical.BasicContingencyManager
    :members:
.. autoclass:: scores.categorical.ThresholdEventOperator
    :members:
.. autoclass:: scores.categorical.EventOperator
    :members:
```

## scores.spatial
```{eval-rst}
.. autoclass:: scores.fast.fss.typing.FssComputeMethod
    :members:
    :member-order: bysource
.. autofunction:: scores.spatial.fss_2d
.. autofunction:: scores.spatial.fss_2d_binary
.. autofunction:: scores.spatial.fss_2d_single_field
```

## scores.stats
```{eval-rst}
.. autofunction:: scores.stats.statistical_tests.diebold_mariano
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

## scores.pandas
```{eval-rst}
.. autofunction:: scores.pandas.continuous.mse
.. autofunction:: scores.pandas.continuous.rmse
.. autofunction:: scores.pandas.continuous.mae
```

## scores.emerging
```{eval-rst}
.. autofunction:: scores.emerging.risk_matrix_score
.. autofunction:: scores.emerging.matrix_weights_to_array
.. autofunction:: scores.emerging.weights_from_warning_scaling
```