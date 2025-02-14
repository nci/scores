# Index of Metrics, Statistical Techniques and Data Processing Tools Included in `scores` 

## Continuous

```{list-table} 
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Additive Bias (Mean Error)
  - [API](api.md#scores.continuous.additive_bias)
  - [Tutorial](project:./tutorials/Additive_and_multiplicative_bias.md)
  - [Mean Error (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#meanerror)
* - Consistent Expectile Score
  - [API](api.md#scores.continuous.consistent_expectile_score)
  - [Tutorial](project:./tutorials/Consistent_Scores.md)
  - [Gneiting (2011)](https://doi.org/10.1198/jasa.2011.r10138); [Taggart (2022)](https://doi.org/10.1002/qj.4206)
* - Consistent Huber Score
  - [API](api.md#scores.continuous.consistent_huber_score)
  - [Tutorial](project:./tutorials/Consistent_Scores.md)
  - [Taggart (2022a)](https://doi.org/10.1002/qj.4206); [Taggart (2022b)](https://doi.org/10.1214/21-EJS1957)
* - Consistent Quantile Score
  - [API](api.md#scores.continuous.consistent_quantile_score)
  - [Tutorial](project:./tutorials/Consistent_Scores.md)
  - [Gneiting (2011)](https://doi.org/10.1198/jasa.2011.r10138); [Taggart (2022)](https://doi.org/10.1002/qj.4206)
* - Flip-Flop Index
  - 
  - 
  - 
* -  
    - Flip-Flop Index
  -
    [API](api.md#scores.continuous.flip_flop_index)
  - 
    [Tutorial](project:./tutorials/Flip_Flop_Index.md)
  - 
    [Griffiths et al. (2019)](https://doi.org/10.1002/met.1732); [Griffiths et al. (2021)](https://doi.org/10.1071/ES21010)
* -  
    - Flip-Flop Index - Proportion Exceeding
  -
    [API](api.md#scores.continuous.flip_flop_index_proportion_exceeding)
  - 
    [Tutorial](project:./tutorials/Flip_Flop_Index.md)
  - 
    [Griffiths et al. (2019)](https://doi.org/10.1002/met.1732); [Griffiths et al. (2021)](https://doi.org/10.1071/ES21010)
* - Interval Score
  - [API](api.md#scores.continuous.interval_score)
  - [Tutorial](project:./tutorials/Quantile_Interval_And_Interval_Score.md)
  - [Gneiting and Raftery (2007) - Section 6.2](https://doi.org/10.1198/016214506000001437)
* - Isotonic Fit, *see Isotonic Regression*
  - &mdash;
  - &mdash;
  - &mdash;
* - Isotonic Regression (Isotonic Fit, Reliability Diagram)
  - [API](api.md#scores.continuous.isotonic_fit)
  - [Tutorial](project:./tutorials/Isotonic_Regression_And_Reliability_Diagrams.md)
  - [de Leeuw et al. (2009)](https://doi.org/10.18637/jss.v032.i05); [Dimitriadis et al. (2020)](https://doi.org/10.1073/pnas.2016191118); [Jordan et al. (2020), version 2](https://doi.org/10.48550/arXiv.1904.04761) 
* - Kling–Gupta Efficiency (KGE)
  - [API](api.md#scores.continuous.kge)
  - [Tutorial](project:./tutorials/Kling_Gupta_Efficiency.md)
  - [Gupta et al. (2009)](https://doi.org/10.1016/j.jhydrol.2009.08.003); [Knoben et al. (2019)](https://doi.org/10.5194/hess-23-4323-2019)    
* - Mean Absolute Error (MAE)
  - [API](api.md#scores.continuous.mae)
  - [Tutorial](project:./tutorials/Mean_Absolute_Error.md)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
* - Mean Elementary Score, *see Murphy Score*
  - &mdash;
  - &mdash;
  - &mdash;
* - Mean Error (Additive Bias)
  - [API](api.md#scores.continuous.mean_error)
  - [Tutorial](project:./tutorials/Additive_and_multiplicative_bias.md)
  - [Mean Error (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#meanerror)
* - Mean Squared Error (MSE)
  - [API](api.md#scores.continuous.mse)
  - [Tutorial](project:./tutorials/Mean_Squared_Error.md)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
* - Multiplicative Bias
  - [API](api.md#scores.continuous.multiplicative_bias)
  - [Tutorial](project:./tutorials/Additive_and_multiplicative_bias.md)
  - [Multiplicative bias (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#multiplicative_bias)
* - Murphy Score (Mean Elementary Score) 
  - 
  - 
  - 
* -  
    - Murphy Score (Mean Elementary Score)
  -
    [API](api.md#scores.continuous.murphy_score)
  - 
    [Tutorial](project:./tutorials/Murphy_Diagrams.md)
  - 
    [Ehm et al. (2016) - Theorem 1](https://doi.org/10.1111/rssb.12154); [Taggart (2022) - Theorem 5.3](https://doi.org/10.1214/21-ejs1957)
* -  
    - Murphy Score (Mean Elementary Score) - Theta Values
  -
    [API](api.md#scores.continuous.murphy_thetas)
  - 
    [Tutorial](project:./tutorials/Murphy_Diagrams.md)
  - 
    [Ehm et al. (2016) - Corollary 2 (p.521)](https://doi.org/10.1111/rssb.12154); [Taggart (2022) - Corollary 5.6](https://doi.org/10.1214/21-ejs1957)
* - Pearson's Correlation Coefficient
  - [API](api.md#scores.continuous.correlation.pearsonr)
  - [Tutorial](project:./tutorials/Pearsons_Correlation.md)
  - [Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
* - Percent Bias (PBIAS)
  - [API](api.md#scores.continuous.pbias)
  - [Tutorial](project:./tutorials/Additive_and_multiplicative_bias.md)
  - [Percent Bias (CRAN hydroGOF)](https://search.r-project.org/CRAN/refmans/hydroGOF/html/pbias.html); [Sorooshian et al. (1993)](https://doi.org/10.1029/92WR02617); [Alfieri et al. (2014)](https://doi.org/10.1016/j.jhydrol.2014.06.035); [Dawson et al. (2007)](https://doi.org/10.1016/j.envsoft.2006.06.008); [Moriasi et al. (2007)](https://doi.org/10.13031/2013.23153)
* - Pinball Loss, *see Quantile Loss*
  - &mdash;
  - &mdash;
  - &mdash;
* - Quantile Interval Score
  - [API](api.md#scores.continuous.quantile_interval_score)
  - [Tutorial](project:./tutorials/Quantile_Interval_And_Interval_Score.md)
  - [Winkler (1972) ](https://doi.org/10.2307/2284720)
* - Quantile Loss (Quantile Score, Pinball Loss)
  - [API](api.md#scores.continuous.quantile_score)
  - [Tutorial](project:./tutorials/Quantile_Loss.md)
  - [Gneiting (2011) - Theorem 9](https://doi.org/10.1198/jasa.2011.r10138)
* - Quantile Score, *see Quantile Loss*
  - &mdash;
  - &mdash;
  - &mdash;
* - Reliability Diagram, *see Isotonic Regression*
  - &mdash;
  - &mdash;
  - &mdash;
* - Root Mean Squared Error (RMSE)
  - [API](api.md#scores.continuous.rmse)
  - [Tutorial](project:./tutorials/Root_Mean_Squared_Error.md)
  - [Wikipedia](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
* - Threshold Weighted Absolute Error
  - [API](api.md#scores.continuous.tw_absolute_error)
  - [Tutorial](project:./tutorials/Threshold_Weighted_Scores.md)
  - [Taggart (2022)](https://doi.org/10.1002/qj.4206)
* - Threshold Weighted Expectile Score
  - [API](api.md#scores.continuous.tw_expectile_score)
  - [Tutorial](project:./tutorials/Threshold_Weighted_Scores.md)
  - [Taggart (2022)](https://doi.org/10.1002/qj.4206)
* - Threshold Weighted Huber Loss
  - [API](api.md#scores.continuous.tw_huber_loss)
  - [Tutorial](project:./tutorials/Threshold_Weighted_Scores.md)
  - [Taggart (2022)](https://doi.org/10.1002/qj.4206)
* - Threshold Weighted Quantile Score
  - [API](api.md#scores.continuous.tw_quantile_score)
  - [Tutorial](project:./tutorials/Threshold_Weighted_Scores.md)
  - [Taggart (2022)](https://doi.org/10.1002/qj.4206)
* - Threshold Weighted Squared Error
  - [API](api.md#scores.continuous.tw_squared_error)
  - [Tutorial](project:./tutorials/Threshold_Weighted_Scores.md)
  - [Taggart (2022)](https://doi.org/10.1002/qj.4206)
```

## Probability

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Brier Score
  - [API](api.md#scores.probability.brier_score)
  - [Tutorial](project:./tutorials/Brier_Score.md)
  - [Brier (1950)](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2)
* - Brier Score for Ensembles
  - [API](api.md#scores.probability.brier_score_for_ensemble)
  - [Tutorial](project:./tutorials/Brier_Score.md)
  - [Ferro (2013)](https://doi.org/10.1002/qj.2270)
* - Continuous Ranked Probability Score (CRPS) for Cumulative Distribution Functions (CDFs)
  -    
  - 
  - 
* -  
    - Adjust Forecast for CRPS
  - [API](api.md#scores.probability.adjust_fcst_for_crps)
  - &mdash;
  - &mdash;
* -  
    - CRPS CDF Brier Decomposition
  - [API](api.md#scores.probability.crps_cdf_brier_decomposition)
  - [Tutorial](project:./tutorials/CRPS_for_CDFs.md)
  - &mdash;        
* -  
    - CRPS for CDFs
  - [API](api.md#scores.probability.crps_cdf)
  - [Tutorial](project:./tutorials/CRPS_for_CDFs.md)
  - [Matheson and Winkler (1976)](https://doi.org/10.1287/mnsc.22.10.1087); [Gneiting and Ranjan (2011)](https://doi.org/10.1198/jbes.2010.08110)
* -  
    - CRPS Step Threshold Weights
  - [API](api.md#scores.probability.crps_step_threshold_weight)
  - &mdash;
  - &mdash;
* - Continuous Ranked Probability Score (CRPS) for Ensembles
  -    
  - 
  -  
* - 
    - CRPS for Ensembles
  - [API](api.md#scores.probability.crps_for_ensemble)   
  - [Tutorial](project:./tutorials/CRPS_for_Ensembles.md)
  - [Ferro (2014)](https://doi.org/10.1002/qj.2270); [Gneiting And Raftery (2007)](https://doi.org/10.1198/016214506000001437); [Zamo and Naveau (2018)](https://doi.org/10.1007/s11004-017-9709-7)
* - 
    - Threshold-Weighted CRPS (twCRPS) for Ensembles
  - [API](api.md#scores.probability.tw_crps_for_ensemble)   
  - [Tutorial](project:./tutorials/Threshold_Weighted_CRPS_for_Ensembles.md)
  - [Allen et al. (2023)](https://doi.org/10.1137/22M1532184); [Allen (2024)](https://doi.org/10.18637/jss.v110.i08)    
* - 
    - Interval-Threshold-Weighted CRPS (twCRPS) for Ensembles
  - [API](api.md#scores.probability.interval_tw_crps_for_ensemble)   
  - [Tutorial](project:./tutorials/Threshold_Weighted_CRPS_for_Ensembles.md)
  - [Allen et al. (2023)](https://doi.org/10.1137/22M1532184); [Allen (2024)](https://doi.org/10.18637/jss.v110.i08) 
* - 
    - Tail-Threshold-Weighted CRPS (twCRPS) for Ensembles
  - [API](api.md#scores.probability.tail_tw_crps_for_ensemble)   
  - [Tutorial](project:./tutorials/Threshold_Weighted_CRPS_for_Ensembles.md)
  - [Allen et al. (2023)](https://doi.org/10.1137/22M1532184); [Allen (2024)](https://doi.org/10.18637/jss.v110.i08)
* - Isotonic Fit, *see Isotonic Regression*
  - &mdash;
  - &mdash;
  - &mdash;
* - Isotonic Regression (Isotonic Fit, Reliability Diagram)
  - [API](api.md#scores.probability.isotonic_fit)
  - [Tutorial](project:./tutorials/Isotonic_Regression_And_Reliability_Diagrams.md)
  - [de Leeuw et al. (2009)](https://doi.org/10.18637/jss.v032.i05); [Dimitriadis et al. (2020)](https://doi.org/10.1073/pnas.2016191118); [Jordan et al. (2020), version 2](https://doi.org/10.48550/arXiv.1904.04761)
* - Mean Elementary Score, *see Murphy Score*
  - &mdash;
  - &mdash;
  - &mdash;
* - Murphy Score (Mean Elementary Score) 
  - 
  - 
  - 
* -  
    - Murphy Score (Mean Elementary Score)
  -
    [API](api.md#scores.probability.murphy_score)
  - 
    [Tutorial](project:./tutorials/Murphy_Diagrams.md)
  - 
    [Ehm et al. (2016) - Theorem 1](https://doi.org/10.1111/rssb.12154); [Taggart (2022) - Theorem 5.3](https://doi.org/10.1214/21-ejs1957)
* -  
    - Murphy Score (Mean Elementary Score) - Theta Values
  -
    [API](api.md#scores.probability.murphy_thetas)
  - 
    [Tutorial](project:./tutorials/Murphy_Diagrams.md)
  - 
    [Ehm et al. (2016) - Corollary 2 (p.521)](https://doi.org/10.1111/rssb.12154); [Taggart (2022) - Corollary 5.6](https://doi.org/10.1214/21-ejs1957)
* - Receiver (Relative) Operating Characteristic (ROC)
  - [API](api.md#scores.probability.roc_curve_data)   
  - [Tutorial](project:./tutorials/ROC.md)
  - [Fawcett and Niculescu-Mizil (2007)](https://doi.org/10.1007/s10994-007-5011-0); [Gneiting and Vogel (2022)](https://doi.org/10.1007/s10994-021-06115-2); [Hand (2009)](https://doi.org/10.1007/s10994-009-5119-5); [Hand and Anagnostopoulos (2013)](https://doi.org/10.1016/j.patrec.2012.12.004)); [Hand and Anagnostopoulos (2023)](https://doi.org/10.1007/s11634-021-00490-3); [Pesce et al. (2010)](https://doi.org/10.1016/j.acra.2010.04.001)
* - Reliability Diagram, *see Isotonic Regression*
  - &mdash;
  - &mdash;
  - &mdash;
```

## Categorical

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Binary Contingency Scores and Binary Contingency Tables
  - [API](api.md#scores.categorical.BinaryContingencyManager); [API](api.md#scores.categorical.BasicContingencyManager)
  - [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - [Methods for dichotomous (yes/no) forecasts](https://www.cawcr.gov.au/projects/verification/#Methods_for_dichotomous_forecasts)
* -  
    - Accuracy (Fraction Correct)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.accuracy)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Accuracy (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#ACC)
* -  
    - Base Rate
  -
    [API](api.md#scores.categorical.BasicContingencyManager.base_rate)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Hogan and Mason (2011)](https://doi.org/10.1002/9781119960003.ch3)
* -  
    - Bias Score (Frequency Bias)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.bias_score)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Bias Score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#BIAS)
* -  
    - Cohen's Kappa (Heidke Skill Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.cohens_kappa)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
     [Heidke Skill Score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HSS) 
* -  
    - Critical Success Index (Threat Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.critical_success_index)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Threat score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#CSI)
* -  
    - Equitable Threat Score (Gilbert Skill Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.equitable_threat_score)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Hogan et al. (2010)](https://doi.org/10.1175/2009WAF2222350.1); [Equitable Threat score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#ETS)
* -  
    - F1 Score
  -
    [API](api.md#scores.categorical.BasicContingencyManager.f1_score)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/F-score)
* -  
    - False Alarm Rate (Probability of False Detection (POFD))
  -
    [API](api.md#scores.categorical.BasicContingencyManager.false_alarm_rate)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Probability of false detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POFD)
* -  
    - False Alarm Ratio (FAR)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.false_alarm_ratio)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [False alarm ratio (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#FAR)
* -  
    - Forecast Rate
  -
    [API](api.md#scores.categorical.BasicContingencyManager.forecast_rate)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Hogan and Mason (2011)](https://doi.org/10.1002/9781119960003.ch3)
* -  
    - Fraction Correct (Accuracy)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.fraction_correct)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Accuracy (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#ACC)
* -  
    - Frequency Bias (Bias Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.frequency_bias)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Bias Score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#BIAS)
* -  
    - Gilbert Skill Score (Equitable Threat Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.gilberts_skill_score)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Hogan et al. (2010)](https://doi.org/10.1175/2009WAF2222350.1); [Equitable Threat score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#ETS) 
* -  
    - Hanssen and Kuipers' Discriminant (Peirce's Skill Score, True Skill Statistic)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.hanssen_and_kuipers_discriminant)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Hanssen and Kuipers discriminant (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HK)
* -  
    - Heidke Skill Score (Cohen's Kappa)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.heidke_skill_score)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
     [Heidke skill score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HSS) 
* -  
    - Hit Rate (True Positive Rate, Probability of Detection (POD), Sensitivity, Recall)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.hit_rate)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* -  
    - Negative Predictive Value
  -
    [API](api.md#scores.categorical.BasicContingencyManager.negative_predictive_value)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values); [Monaghan et al. (2021)](https://doi.org/10.3390/medicina57050503)
* -  
    - Odds Ratio
  -
    [API](api.md#scores.categorical.BasicContingencyManager.odds_ratio)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Stephenson (2000)](https://doi.org/10.1175/1520-0434(2000)015<0221:UOTORF>2.0.CO;2)
* -  
    - Odds Ratio Skill Score (Yule's Q)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.odds_ratio_skill_score)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Stephenson (2000)](https://doi.org/10.1175/1520-0434(2000)015<0221:UOTORF>2.0.CO;2)
* -  
    - Peirce's Skill Score (True Skill Statistic, Hanssen and Kuipers' Discriminant)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.peirce_skill_score)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Peirce (1884)](https://doi.org/10.1126/science.ns-4.93.453.b); [Hanssen and Kuipers discriminant (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HK)
* -  
    - Positive Predictive Value (Success Ratio, Precision)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.positive_predictive_value)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values); [Success ratio (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#SR); [Monaghan et al. (2021)](https://doi.org/10.3390/medicina57050503)
* -  
    - Precision (Success Ratio, Positive Predictive Value)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.precision)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall); [Success ratio (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#SR)
* -  
    - Probability of Detection (POD) (Hit Rate, True Positive Rate, Sensitivity, Recall)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.probability_of_detection)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* -  
    - Probability of False Detection (POFD) (False Alarm Rate)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.probability_of_false_detection)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Probability of false detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POFD)
* -  
    - Recall (Hit Rate, Probability of Detection (POD), True Positive Rate, Sensitivity)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.recall)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall); [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* -  
    - Sensitivity (Hit Rate, Probability of Detection (POD), True Positive Rate, Recall)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.sensitivity)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity); [Monaghan et al. (2021)](https://doi.org/10.3390/medicina57050503)
* -  
    - Specificity (True Negative Rate)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.specificity)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity); [Monaghan et al. (2021)](https://doi.org/10.3390/medicina57050503)
* -  
    - Success Ratio (Precision, Positive Predictive Value)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.success_ratio)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Success ratio (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#SR)    
* -  
    - Symmetric Extremal Dependence Index (SEDI)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.symmetric_extremal_dependence_index)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Ferro and Stephenson (2011)](https://doi.org/10.1175/WAF-D-10-05030.1)
* -  
    - Threat Score (Critical Success Index)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.threat_score)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Threat score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#CSI)    
* -  
    - True Negative Rate (Specificity)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.true_negative_rate)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* -  
    - True Positive Rate (Hit Rate, Probability of Detection (POD), Sensitivity, Recall)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.true_positive_rate)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* -  
    - True Skill Statistic (Peirce's Skill Score, Hanssen and Kuipers' Discriminant)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.true_skill_statistic)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Hanssen and Kuipers discriminant (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HK)
* -  
    - Yule's Q (Odds Ratio Skill Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.yules_q)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    [Stephenson (2000)](https://doi.org/10.1175/1520-0434(2000)015<0221:UOTORF>2.0.CO;2) 
* - FIxed Risk Multicategorical (FIRM)
  - [API](api.md#scores.categorical.firm)
  - [Tutorial](project:./tutorials/FIRM.md)
  - [Taggart et al. (2022)](https://doi.org/10.1002/qj.4266)
* - POD - implementation as used in ROC (***NOTE:*** **Please use contingency table classes instead, this API may be removed in future**)
  - [API](api.md#scores.categorical.probability_of_detection)   
  - [Tutorial](project:./tutorials/ROC.md)
  - [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* - POFD - implementation as used in ROC (***NOTE:*** **Please use contingency table classes instead, this API may be removed in future**)
  - [API](api.md#scores.categorical.probability_of_false_detection)
  - [Tutorial](project:./tutorials/ROC.md)
  - [Probability of false detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POFD)
* - Threshold Event Operator
  - [API](api.md#scores.categorical.ThresholdEventOperator)
  - [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - &mdash;
* -  
    - Make Contingency Manager
  -
    [API](api.md#scores.categorical.ThresholdEventOperator.make_contingency_manager)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    &mdash;
* -  
    - Make Event Tables
  -
    [API](api.md#scores.categorical.ThresholdEventOperator.make_event_tables)
  - 
    [Tutorial](project:./tutorials/Binary_Contingency_Scores.md)
  - 
    &mdash;    
```

## Spatial

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Fractions Skill Score (FSS)
  - 
  - 
  - 
* -  
    - FSS - 2D
  -
    [API](api.md#scores.spatial.fss_2d)
  - 
    [Tutorial](project:./tutorials/Fractions_Skill_Score.md)
  - 
    [Roberts and Lean (2008)](https://doi.org/10.1175/2007mwr2123.1); [Mittermaier (2021)](https://doi.org/10.1175/mwr-d-18-0106.1)
* -  
    - FSS - 2D Binary
  -
    [API](api.md#scores.spatial.fss_2d_binary)
  - 
    [Tutorial](project:./tutorials/Fractions_Skill_Score.md)
  - 
    &mdash;
* -  
    - FSS - 2D Single Field
  -
    [API](api.md#scores.spatial.fss_2d_single_field)
  - 
    [Tutorial](project:./tutorials/Fractions_Skill_Score.md)
  - 
    [Roberts and Lean (2008)](https://doi.org/10.1175/2007mwr2123.1); [Faggian et al. (2015)](https://doi.org/10.54302/mausam.v66i3.555)
```

## Statistical Tests

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Diebold Mariano (with the Harvey et al. 1997 and the Hering and Genton 2011 modifications)
  - [API](api.md#scores.stats.statistical_tests.diebold_mariano)
  - [Tutorial](project:./tutorials/Diebold_Mariano_Test_Statistic.md)
  - [Diebold and Mariano (1995)](https://doi.org/10.1080/07350015.1995.10524599); [Harvey et al. (1997)](https://doi.org/10.1016/S0169-2070(96)00719-4); [Hering and Genton (2011)](https://doi.org/10.1198/TECH.2011.10136)
```

## Processing (tools for preparing data)

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Utilised For
* - Binary Discretise
  - [API](api.md#scores.processing.binary_discretise)   
  - Receiver (Relative) Operating Characteristic (ROC)
* - Binary Discretise Proportion
  - [API](api.md#scores.processing.binary_discretise_proportion)    
  - Flip-Flop Index
* - Block Bootstrap
  - [API](api.md#scores.processing.block_bootstrap)
  - Confidence intervals. See [tutorial](project:./tutorials/Block_Bootstrapping.md)
* - Broadcast and Match Not-a-Number (NaN)
  - [API](api.md#scores.processing.broadcast_and_match_nan)   
  - Murphy Score (Mean Elementary Score)
* - Comparative Discretise
  - [API](api.md#scores.processing.comparative_discretise)   
  - Receiver (Relative) Operating Characteristic (ROC)
* - Cumulative Distribution Functions (CDFs)
  - 
  - 
* -  
    - Add Thresholds
  -
    [API](api.md#scores.processing.cdf.add_thresholds)
  - 
    Continuous Ranked Probability Score (CRPS) for CDFs; CRPS CDF Brier Decomposition
* -  
    - CDF Envelope
  -
    [API](api.md#scores.processing.cdf.cdf_envelope)
  - 
    Adjust Forecast for CRPS     
* -  
    - Decreasing CDFs
  -
    [API](api.md#scores.processing.cdf.decreasing_cdfs)
  - 
    Adjust Forecast for CRPS
* -  
    - Fill CDF
  -
    [API](api.md#scores.processing.cdf.fill_cdf)
  - 
    CRPS for CDFs; CRPS CDF Brier Decomposition
* -  
    - Integrate Square Piecewise Linear
  -
    [API](api.md#scores.processing.cdf.integrate_square_piecewise_linear)
  - 
    CRPS for CDFs 
* -  
    - Observed CDF
  -
    [API](api.md#scores.processing.cdf.observed_cdf)
  - 
    CRPS for CDFs; CRPS CDF Brier Decomposition
* -  
    - Propagate Not-a-Number (NaN) 
  -
    [API](api.md#scores.processing.cdf.propagate_nan)
  - 
    Adjust Forecast for CRPS; CRPS CDF Brier Decomposition; CRPS for CDFs   
* -  
    - Round Values 
  -
    [API](api.md#scores.processing.cdf.round_values)
  - 
    CRPS for CDFs; CRPS CDF Brier Decomposition
* - Isotonic Fit, *see Isotonic Regression*
  - &mdash;
  - &mdash;
* - Isotonic Regression (Isotonic Fit, Reliability Diagram)
  - [API](api.md#scores.processing.isotonic_fit)
  - See "Isotonic Regression (Isotonic Fit, Reliability Diagram)" entries in [Continuous](#continuous) and [Probability](#probability)
* - Proportion Exceeding
  - [API](api.md#scores.processing.proportion_exceeding)   
  - Flip-Flop Index
* - Reliability Diagram, *see Isotonic Regression*
  - &mdash;
  - &mdash;   
```

## Pandas

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Mean Absolute Error
  - [API](api.md#scores.pandas.continuous.mae)
  - [Tutorial](project:./tutorials/Pandas_API.md)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
* - Mean Squared Error
  - [API](api.md#scores.pandas.continuous.mse)  
  - [Tutorial](project:./tutorials/Pandas_API.md)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
* - Root Mean Squared Error
  - [API](api.md#scores.pandas.continuous.rmse)   
  - [Tutorial](project:./tutorials/Pandas_API.md)
  - [Wikipedia](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
```

## Emerging

```{Caution} 
  This section of the API contains implementations of novel metrics that are still undergoing mathematical peer review. These implementations may change in line with the peer review process.
```

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Risk Matrix Score
  - 
  - 
  - 
* -  
    - Risk Matrix Score
  - [API](api.md#scores.emerging.risk_matrix_score)
  - [Tutorial](project:./tutorials/Risk_Matrix_Score.md)
  - Taggart, R. J., & Wilke, D. J. (2025). Warnings based on risk matrices: a coherent framework with consistent evaluation. https://doi.org/10.48550/arXiv.2502.08891

* -  
    - Risk Matrix Score - Matrix Weights to Array
  - [API](api.md#scores.emerging.matrix_weights_to_array)
  - [Tutorial](project:./tutorials/Risk_Matrix_Score.md)
  - Taggart, R. J., & Wilke, D. J. (2025). Warnings based on risk matrices: a coherent framework with consistent evaluation. https://doi.org/10.48550/arXiv.2502.08891

* -  
    - Risk Matrix Score - Warning Scaling to Weight Array
  - [API](api.md#scores.emerging.weights_from_warning_scaling)
  - [Tutorial](project:./tutorials/Risk_Matrix_Score.md)
  - Taggart, R. J., & Wilke, D. J. (2025). Warnings based on risk matrices: a coherent framework with consistent evaluation. https://doi.org/10.48550/arXiv.2502.08891

```
