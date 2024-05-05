# Index of Metrics, statistical techniques and data processing tools included in `scores` 

****NOTE:**** TBA = To be added - these are things either already merged into development or currently in a pull request 

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
  - [Mean Error (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#meanerror)
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
* - Isotonic Fit, see Isotonic Regression
  - ...
  - ...
  - ...
* - Isotonic Regression (Isotonic Fit, Relability Diagram)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.continuous.isotonic_fit)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Isotonic_Regression.html)
  - [de Leeuw et al. (2009)](https://doi.org/10.18637/jss.v032.i05); [Dimitriadis et al. (2020)](https://doi.org/10.1073/pnas.2016191118); [Jordan et al. (2020), version 2](https://doi.org/10.48550/arXiv.1904.04761)   
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
  - [Multiplicative bias (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#multiplicative_bias)
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
  - [Gneiting (2011) - Theorem 9](https://doi.org/10.1198/jasa.2011.r10138)
* - Quantile Score, see Quantile Loss
  - ...
  - ...
  - ...
* - Relability Diagram, see Isotonic Regression
  - ...
  - ...
  - ...
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
  - [Brier (1950)](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2)
* - Continuous Ranked Probability Score (CRPS) for cumulative distribution functions (CDFs)
  -    
  - TBA
  -
* - Continuous Ranked Probability Score (CRPS) for ensembles
  -    
  - TBA
  -
* - Isotonic Fit, see Isotonic Regression
  - ...
  - ...
  - ...
* - Isotonic Regression (Isotonic Fit, Relability Diagram)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.continuous.isotonic_fit)
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/Isotonic_Regression.html)
  - [de Leeuw et al. (2009)](https://doi.org/10.18637/jss.v032.i05); [Dimitriadis et al. (2020)](https://doi.org/10.1073/pnas.2016191118); [Jordan et al. (2020), version 2](https://doi.org/10.48550/arXiv.1904.04761)
* - Mean Elementary Score, see Murphy Score
  - ...
  - ...
  - ...
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
* - Receiver (Relative) Operating Characteristic (ROC)
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.probability.roc_curve_data)   
  - [Tutorial](https://scores.readthedocs.io/en/latest/tutorials/ROC.html)
  - [Fawcett and Niculescu-Mizil (2007)](https://doi.org/10.1007/s10994-007-5011-0); [Gneiting and Vogel (2022)](https://doi.org/10.1007/s10994-021-06115-2); [Hand (2009)](https://doi.org/10.1007/s10994-009-5119-5); [Hand and Anagnostopoulos (2013)](https://doi.org/10.1016/j.patrec.2012.12.004)); [Hand and Anagnostopoulos (2023)](https://doi.org/10.1007/s11634-021-00490-3); [Pesce et al. (2010)](https://doi.org/10.1016/j.acra.2010.04.001)
* - Relability Diagram, see Isotonic Regression
  - ...
  - ...
  - ...  
```

## Categorical

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
    - Accuracy (Fraction Correct)
  -
    TBA
  - 
    TBA
  - 
    [Accuracy (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#ACC)
* -  
    - False Alarm Rate (Probability of False Detection (POFD))
  -
    TBA
  - 
    TBA
  - 
    [Probability of false detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POFD)
* -  
    - Frequency Bias (Bias Score)
  -
    TBA
  - 
    TBA
  - 
    [Bias Score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#BIAS)
* -  
    - Hit Rate (True Positive Rate, Probability of Detection (POD), Sensitivity)
  -
    TBA
  - 
    TBA
  - 
    [Probability of detection (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#POD)
* -  
    - Peirce's Skill Score (True Skill Statistic, Hannssen and Kuipers' Discriminant)
  -
    TBA
  - 
    TBA
  - 
    [Hanssen and Kuipers discriminant (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#HK)
* -  
    - Specificity
  -
    TBA
  - 
    TBA
  - 
    Ref 
* -  
    - Success Ratio
  -
    TBA
  - 
    TBA
  - 
    [Success ratio (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#SR)    
* -  
    - Threat Score (Critical Success Index)
  -
    TBA
  - 
    TBA
  - 
    [Threat score (WWRP/WGNE Joint Working Group on Forecast Verification Research)](https://www.cawcr.gov.au/projects/verification/#CSI)    
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
  - [Diebold and Mariano (1995)](https://doi.org/10.1080/07350015.1995.10524599); [Harvey et al. (1997)](https://doi.org/10.1016/S0169-2070(96)00719-4); [Hering and Genton (2011)](https://doi.org/10.1198/TECH.2011.10136)
```

## Processing (tools for preparing data)

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

## Pandas

```{list-table}
:header-rows: 1

* - Name (Alphabetical order)
  - API
  - Tutorial
  - Reference(s)
* - Mean Absolute Error
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.pandas.continuous.mae)
  - TBA
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
* - Mean Squared Error
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.pandas.continuous.mse)  
  - TBA
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
* - Root Mean Squared Error
  - [API](https://scores.readthedocs.io/en/latest/api.html#scores.pandas.continuous.rmse)   
  - TBA
  - [Wikipedia](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
```


