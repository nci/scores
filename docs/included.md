# Metrics, statistical techniques and data processing tools included in `scores` 

## Continuous

```{list-table} 
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Additive Bias (Mean Error)
  - TBA
  - TBA
  - [Forecast Verification Site by WWRP/WGNE Joint Working Group on Forecast Verification Research](https://www.cawcr.gov.au/projects/verification/)
* - Flip-Flop Index
  - 
  - 
  - 
* -  
    - Flip-Flop Index
  -
    [API](https://scores.readthedocs.io/en/latest/api.html#scores.continuous.flip_flop_index)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Flip_Flop_Index.html)
  - 
    [Griffiths et al. (2019)](https://doi.org/10.1002/met.1732); [Griffiths et al. (2021)](https://doi.org/10.1071/ES21010)
* -  
    - Flip-Flop Index - proportion exceeding
  -
    [API](https://scores.readthedocs.io/en/latest/api.html#scores.continuous.flip_flop_index_proportion_exceeding)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Flip_Flop_Index.html)
  - 
    [Griffiths et al. (2019)](https://doi.org/10.1002/met.1732); [Griffiths et al. (2021)](https://doi.org/10.1071/ES21010)
* - Mean Absolute Error (MAE)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mae)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Mean_Absolute_Error.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
* - Mean Elementary Score, see Murphy Score
  - ...
  - ...
  - ...
* - Mean Error, see Additive Bias
  - ...
  - ...
  - ...
* - Mean Squared Error (MSE)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mse)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Mean_Squared_Error.html)
  - Missing Ref
* - Multiplicative Bias
  - TBA
  - TBA
  - [Forecast Verification Site by WWRP/WGNE Joint Working Group on Forecast Verification Research](https://www.cawcr.gov.au/projects/verification/)
* - Murphy Score (Mean Elementary Score) 
  - 
  - 
  - 
* -  
    - Murphy Score (Mean Elementary Score)
  -
    [API](https://scores.readthedocs.io/en/latest/api.html#scores.continuous.murphy_score)
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Murphy_Diagrams.html)
  - 
    [Ehm et al. (2016) - Theorem 1](https://www.jstor.org/stable/24775351); [Taggart (2022) - Theorem 5.3](https://doi.org/10.1214/21-ejs1957)
* -  
    - Murphy Score (Mean Elementary Score) - theta values
  -
    TBA
  - 
    [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Murphy_Diagrams.html)
  - 
    [Ehm et al. (2016) - Corollary 2 (p.521)](https://www.jstor.org/stable/24775351); [Taggart (2022) - Corollary 5.6](https://doi.org/10.1214/21-ejs1957)
* - Pearson's Correlation Coefficient
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.continuous.correlation)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Pearsons_Correlation.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
* - Quantile Loss (Quantile Score)
  - TBA
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Quantile_Loss.html)
  - [Gneiting (2011)](https://doi.org/10.1198/jasa.2011.r10138)
* - Root Mean Squared Error (RMSE)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.continuous.rmse)
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
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.probability.brier_score)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Brier_Score.html)
  - [Brier 1950](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2)
* - Continuous Ranked Probability Score (CRPS) for cumulative distribution functions (CDFs)
  -    
  - 
  -
* - Continuous Ranked Probability Score (CRPS) for ensembles
  -    
  - 
  -
* - Receiver (Relative) Operating Characteristic (ROC)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.probability.roc_curve_data)   
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/ROC.html)
  -
```

## Categorical

#### Version One: Including what is currently in main (i.e. no binary contingency table class)
```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - FIxed Risk Multicategorical (FIRM)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.categorical.firm)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/FIRM.html)
  - [Taggart et al. (2022)](https://doi.org/10.1002/qj.4266)
* - POD - implementation as used in ROC
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.categorical.probability_of_detection)   
  - Not available
  - Missing Ref
* - POFD - implementation as used in ROC
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.categorical.probability_of_false_detection)
  - Not available
  - Missing Ref
```

#### Version two: based on what is in main plus the current binary contingency table class work
```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Binary Contingency Table Class (Binary Contingency Scores)
  - 
  - 
  - 
* -  
    - Accuracy (fraction correct)
  -
    TBA
  - 
    TBA
  - 
    [Forecast Verification Site by WWRP/WGNE Joint Working Group on Forecast Verification Research](https://www.cawcr.gov.au/projects/verification/)
* -  
    - False alarm rate (probability of false detection (POFD))
  -
    TBA
  - 
    TBA
  - 
    [Forecast Verification Site by WWRP/WGNE Joint Working Group on Forecast Verification Research](https://www.cawcr.gov.au/projects/verification/)
* -  
    - Frequency bias (bias score)
  -
    TBA
  - 
    TBA
  - 
    [Forecast Verification Site by WWRP/WGNE Joint Working Group on Forecast Verification Research](https://www.cawcr.gov.au/projects/verification/)
* -  
    - Hit rate (true positive rate, probability of detection (POD), sensitivity)
  -
    TBA
  - 
    TBA
  - 
    [Forecast Verification Site by WWRP/WGNE Joint Working Group on Forecast Verification Research](https://www.cawcr.gov.au/projects/verification/)
* -  
    - Peirce's skill score (true skill statistic, Hannssen and Kuipers' discriminant)
  -
    TBA
  - 
    TBA
  - 
    [Forecast Verification Site by WWRP/WGNE Joint Working Group on Forecast Verification Research](https://www.cawcr.gov.au/projects/verification/)
* -  
    - Specificity
  -
    TBA
  - 
    TBA
  - 
    Ref 
* -  
    - Success ratio
  -
    TBA
  - 
    TBA
  - 
    [Forecast Verification Site by WWRP/WGNE Joint Working Group on Forecast Verification Research](https://www.cawcr.gov.au/projects/verification/)    
* -  
    - Threat score (critical success index)
  -
    TBA
  - 
    TBA
  - 
    [Forecast Verification Site by WWRP/WGNE Joint Working Group on Forecast Verification Research](https://www.cawcr.gov.au/projects/verification/)    
* - FIxed Risk Multicategorical (FIRM)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.categorical.firm)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/FIRM.html)
  - [Taggart et al. (2022)](https://doi.org/10.1002/qj.4266)
* - POD - implementation as used in ROC
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.categorical.probability_of_detection)   
  - Not available
  - Missing Ref
* - POFD - implementation as used in ROC
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.categorical.probability_of_false_detection)
  - Not available
  - Missing Ref
```

## Statistical Tests

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Diebold Mariano (with the Harvey et al. 1997 and the Hering and Genton 2011 modifications)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.stats.statistical_tests.diebold_mariano)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Diebold_Mariano_Test_Statistic.html)
  - [Diebold and Mariano (1995)](https://doi.org/10.1080/07350015.1995.10524599);[Harvey et al. (1997)](https://doi.org/10.1016/S0169-2070(96)00719-4);[Hering and Genton (2011)](https://doi.org/10.1198/TECH.2011.10136)
```

## Other: Processing (Pre-processing tools for preparing data)

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Binary Discretise
  -    
  - 
  -
* - Broadcast and Match NaN
  -    
  - 
  -
* - Comparative Discretise
  -    
  - 
  -
* - Isotonic Regression (Isotonic Fit, Reliability Diagram)
  - 
  - 
  - 
* - Proportion Exceeding
  -    
  - 
  -   
```
