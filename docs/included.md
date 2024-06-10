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
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Additive_and_multiplicative_bias.html)
  - [Mean Error (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#meanerror)
* - Flip-Flop Index
  - 
  - 
  - 
* -  
    - Flip-Flop Index
  -
    [API](api.md#scores.continuous.flip_flop_index)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Flip_Flop_Index.html)
  - 
    [Griffiths et al. (2019)](https://doi.org/10.1002/met.1732); [Griffiths et al. (2021)](https://doi.org/10.1071/ES21010)
* -  
    - Flip-Flop Index - Proportion Exceeding
  -
    [API](api.md#scores.continuous.flip_flop_index_proportion_exceeding)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Flip_Flop_Index.html)
  - 
    [Griffiths et al. (2019)](https://doi.org/10.1002/met.1732); [Griffiths et al. (2021)](https://doi.org/10.1071/ES21010)
* - Isotonic Fit, *see Isotonic Regression*
  - &mdash;
  - &mdash;
  - &mdash;
* - Isotonic Regression (Isotonic Fit, Reliability Diagram)
  - [API](api.md#scores.continuous.isotonic_fit)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Isotonic_Regression_And_Reliability_Diagrams.html)
  - [de Leeuw et al. (2009)](https://doi.org/10.18637/jss.v032.i05); [Dimitriadis et al. (2020)](https://doi.org/10.1073/pnas.2016191118); [Jordan et al. (2020), version 2](https://doi.org/10.48550/arXiv.1904.04761)   
* - Mean Absolute Error (MAE)
  - [API](api.md#scores.continuous.mae)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Mean_Absolute_Error.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
* - Mean Elementary Score, *see Murphy Score*
  - &mdash;
  - &mdash;
  - &mdash;
* - Mean Error (Additive Bias)
  - [API](api.md#scores.continuous.mean_error)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Additive_and_multiplicative_bias.html)
  - [Mean Error (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#meanerror)
* - Mean Squared Error (MSE)
  - [API](api.md#scores.continuous.mse)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Mean_Squared_Error.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
* - Multiplicative Bias
  - [API](api.md#scores.continuous.multiplicative_bias)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Additive_and_multiplicative_bias.html)
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
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Murphy_Diagrams.html)
  - 
    [Ehm et al. (2016) - Theorem 1](https://doi.org/10.1111/rssb.12154); [Taggart (2022) - Theorem 5.3](https://doi.org/10.1214/21-ejs1957)
* -  
    - Murphy Score (Mean Elementary Score) - Theta Values
  -
    [API](api.md#scores.continuous.murphy_thetas)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Murphy_Diagrams.html)
  - 
    [Ehm et al. (2016) - Corollary 2 (p.521)](https://doi.org/10.1111/rssb.12154); [Taggart (2022) - Corollary 5.6](https://doi.org/10.1214/21-ejs1957)
* - Pearson's Correlation Coefficient
  - [API](api.md#scores.continuous.correlation)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Pearsons_Correlation.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
* - Pinball Loss, *see Quantile Loss*
  - &mdash;
  - &mdash;
  - &mdash;
* - Quantile Loss (Quantile Score, Pinball Loss)
  - [API](api.md#scores.continuous.quantile_score)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Quantile_Loss.html)
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
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Root_Mean_Squared_Error.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
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
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Brier_Score.html)
  - [Brier (1950)](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2)
* - Continuous Ranked Probability Score (CRPS) for Cumulative Distribution Functions (CDFs)
  -    
  - 
  -
* -  
    - CRPS for CDFs
  -
    [API](api.md#scores.probability.crps_cdf)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/CRPS_for_CDFs.html)
  - 
    [Matheson and Winkler (1976)](https://doi.org/10.1287/mnsc.22.10.1087); [Gneiting and Ranjan (2011)](https://doi.org/10.1198/jbes.2010.08110)
* -  
    - Adjust Forecast for CRPS
  -
    [API](api.md#scores.probability.adjust_fcst_for_crps)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/CRPS_for_CDFs.html)
  - 
    &mdash;
* -  
    - CRPS CDF Brier Decomposition
  -
    [API](api.md#scores.probability.crps_cdf_brier_decomposition)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/CRPS_for_CDFs.html)
  - 
    &mdash;    
* - Continuous Ranked Probability Score (CRPS) for Ensembles
  - [API](api.md#scores.probability.crps_for_ensemble)   
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/CRPS_for_Ensembles.html)
  - [Ferro (2014)](https://doi.org/10.1002/qj.2270); [Gneiting And Raftery (2007)](https://doi.org/10.1198/016214506000001437); [Zamo and Naveau (2018)](https://doi.org/10.1007/s11004-017-9709-7)
* - Isotonic Fit, *see Isotonic Regression*
  - &mdash;
  - &mdash;
  - &mdash;
* - Isotonic Regression (Isotonic Fit, Reliability Diagram)
  - [API](api.md#scores.probability.isotonic_fit)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Isotonic_Regression_And_Reliability_Diagrams.html)
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
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Murphy_Diagrams.html)
  - 
    [Ehm et al. (2016) - Theorem 1](https://doi.org/10.1111/rssb.12154); [Taggart (2022) - Theorem 5.3](https://doi.org/10.1214/21-ejs1957)
* -  
    - Murphy Score (Mean Elementary Score) - Theta Values
  -
    [API](api.md#scores.probability.murphy_thetas)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Murphy_Diagrams.html)
  - 
    [Ehm et al. (2016) - Corollary 2 (p.521)](https://doi.org/10.1111/rssb.12154); [Taggart (2022) - Corollary 5.6](https://doi.org/10.1214/21-ejs1957)
* - Receiver (Relative) Operating Characteristic (ROC)
  - [API](api.md#scores.probability.roc_curve_data)   
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/ROC.html)
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
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - [Methods for dichotomous (yes/no) forecasts](https://www.cawcr.gov.au/projects/verification/#Methods_for_dichotomous_forecasts)
* -  
    - Accuracy (Fraction Correct)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.accuracy)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Accuracy (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#ACC)
* -  
    - Base Rate
  -
    [API](api.md#scores.categorical.BasicContingencyManager.base_rate)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Hogan and Mason (2011)](https://doi.org/10.1002/9781119960003.ch3)
* -  
    - Bias Score (Frequency Bias)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.bias_score)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Bias Score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#BIAS)
* -  
    - Cohen's Kappa (Heidke Skill Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.cohens_kappa)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
     [Heidke Skill Score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HSS) 
* -  
    - Critical Success Index (Threat Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.critical_success_index)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Threat score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#CSI)
* -  
    - Equitable Threat Score (Gilbert Skill Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.equitable_threat_score)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Hogan et al. (2010)](https://doi.org/10.1175/2009WAF2222350.1); [Equitable Threat score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#ETS)
* -  
    - F1 Score
  -
    [API](api.md#scores.categorical.BasicContingencyManager.f1_score)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/F-score)
* -  
    - False Alarm Rate (Probability of False Detection (POFD))
  -
    [API](api.md#scores.categorical.BasicContingencyManager.false_alarm_rate)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Probability of false detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POFD)
* -  
    - False Alarm Ratio (FAR)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.false_alarm_ratio)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [False alarm ratio (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#FAR)
* -  
    - Forecast Rate
  -
    [API](api.md#scores.categorical.BasicContingencyManager.forecast_rate)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Hogan and Mason (2011)](https://doi.org/10.1002/9781119960003.ch3)
* -  
    - Fraction Correct (Accuracy)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.fraction_correct)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Accuracy (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#ACC)
* -  
    - Frequency Bias (Bias Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.frequency_bias)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Bias Score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#BIAS)
* -  
    - Gilbert Skill Score (Equitable Threat Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.gilberts_skill_score)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Hogan et al. (2010)](https://doi.org/10.1175/2009WAF2222350.1); [Equitable Threat score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#ETS) 
* -  
    - Hanssen and Kuipers' Discriminant (Peirce's Skill Score, True Skill Statistic)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.hanssen_and_kuipers_discriminant)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Hanssen and Kuipers discriminant (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HK)
* -  
    - Heidke Skill Score (Cohen's Kappa)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.heidke_skill_score)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
     [Heidke skill score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HSS) 
* -  
    - Hit Rate (True Positive Rate, Probability of Detection (POD), Sensitivity, Recall)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.hit_rate)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* -  
    - Odds Ratio
  -
    [API](api.md#scores.categorical.BasicContingencyManager.odds_ratio)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Stephenson (2000)](https://doi.org/10.1175/1520-0434(2000)015<0221:UOTORF>2.0.CO;2)
* -  
    - Odds Ratio Skill Score (Yule's Q)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.odds_ratio_skill_score)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Stephenson (2000)](https://doi.org/10.1175/1520-0434(2000)015<0221:UOTORF>2.0.CO;2)
* -  
    - Peirce's Skill Score (True Skill Statistic, Hanssen and Kuipers' Discriminant)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.peirce_skill_score)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Peirce (1884)](https://doi.org/10.1126/science.ns-4.93.453.b); [Hanssen and Kuipers discriminant (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HK)
* -  
    - Precision (Success Ratio)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.precision)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall); [Success ratio (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#SR)
* -  
    - Probability of Detection (POD) (Hit Rate, True Positive Rate, Sensitivity, Recall)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.probability_of_detection)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* -  
    - Probability of False Detection (POFD) (False Alarm Rate)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.probability_of_false_detection)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Probability of false detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POFD)
* -  
    - Recall (Hit Rate, Probability of Detection (POD), True Positive Rate, Sensitivity)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.recall)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall); [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* -  
    - Symmetric Extremal Dependence Index (SEDI)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.symmetric_extremal_dependence_index)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Ferro and Stephenson (2011)](https://doi.org/10.1175/WAF-D-10-05030.1)
* -  
    - Sensitivity (Hit Rate, Probability of Detection (POD), True Positive Rate, Recall)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.sensitivity)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* -  
    - Specificity (True Negative Rate)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.specificity)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* -  
    - Success Ratio (Precision)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.success_ratio)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Success ratio (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#SR)    
* -  
    - Threat Score (Critical Success Index)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.threat_score)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Threat score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#CSI)    
* -  
    - True Negative Rate (Specificity)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.true_negative_rate)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* -  
    - True Positive Rate (Hit Rate, Probability of Detection (POD), Sensitivity, Recall)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.true_positive_rate)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* -  
    - True Skill Statistic (Peirce's Skill Score, Hanssen and Kuipers' Discriminant)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.true_skill_statistic)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Hanssen and Kuipers discriminant (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HK)
* -  
    - Yule's Q (Odds Ratio Skill Score)
  -
    [API](api.md#scores.categorical.BasicContingencyManager.yules_q)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    [Stephenson (2000)](https://doi.org/10.1175/1520-0434(2000)015<0221:UOTORF>2.0.CO;2) 
* - FIxed Risk Multicategorical (FIRM)
  - [API](api.md#scores.categorical.firm)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/FIRM.html)
  - [Taggart et al. (2022)](https://doi.org/10.1002/qj.4266)
* - POD - implementation as used in ROC (***NOTE:*** **Please use contingency table classes instead, this API may be removed in future**)
  - [API](api.md#scores.categorical.probability_of_detection)   
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/ROC.html)
  - [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* - POFD - implementation as used in ROC (***NOTE:*** **Please use contingency table classes instead, this API may be removed in future**)
  - [API](api.md#scores.categorical.probability_of_false_detection)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/ROC.html)
  - [Probability of false detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POFD)
* - Threshold Event Operator
  - [API](api.md#scores.categorical.ThresholdEventOperator)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - &mdash;
* -  
    - Make Contingency Manager
  -
    [API](api.md#scores.categorical.ThresholdEventOperator.make_contingency_manager)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
  - 
    &mdash;
* -  
    - Make Event Tables
  -
    [API](api.md#scores.categorical.ThresholdEventOperator.make_event_tables)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Binary_Contingency_Scores.html)
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
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Fractions_Skill_Score.html)
  - 
    [Roberts and Lean (2008)](https://doi.org/10.1175/2007mwr2123.1); [Mittermaier (2021)](https://doi.org/10.1175/mwr-d-18-0106.1)
* -  
    - FSS - 2D Binary
  -
    [API](api.md#scores.spatial.fss_2d_binary)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Fractions_Skill_Score.html)
  - 
    &mdash;
* -  
    - FSS - 2D Single Field
  -
    [API](api.md#scores.spatial.fss_2d_single_field)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Fractions_Skill_Score.html)
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
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Diebold_Mariano_Test_Statistic.html)
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
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Pandas_API.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
* - Mean Squared Error
  - [API](api.md#scores.pandas.continuous.mse)  
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Pandas_API.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
* - Root Mean Squared Error
  - [API](api.md#scores.pandas.continuous.rmse)   
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Pandas_API.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
```


