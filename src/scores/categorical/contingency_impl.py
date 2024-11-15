"""
This foundation class provides an underpinning data structure which can be used for
contingency tables of various kinds, also known as a confusion matrix.

It allows the careful setting out of when and where forecasts closely match
observations.

The binary contingency table captures true positives (hits), true negatives (correct negatives),
false positives (false alarms) and false negatives (misses).

The process of deriving a contingency table relies on the forecast data, the observation
data, and a matching or event operator. The event operator will produce the category from the
forecast and the observed data. Specific use cases may well be supported by a case-specific
matching operator.

Scores supports complex, weighted, multi-dimensional data, including in contingency tables.

Users can supply their own event operators to the top-level module functions.
"""

# pylint: disable=too-many-lines

import operator
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import xarray as xr

import scores.utils
from scores.typing import FlexibleArrayType, FlexibleDimensionTypes

DEFAULT_PRECISION = 8


class BasicContingencyManager:  # pylint: disable=too-many-public-methods
    """
    See https://scores.readthedocs.io/en/stable/tutorials/Binary_Contingency_Scores.html for
    a detailed walkthrough showing the use of this class in practice.

    A BasicContingencyManager object provides the scoring functions which are calculated
    from a contingency table. A BasicContingencyManager can be efficiently and repeatedly
    queried for a wide variety of scores.

    A BasicContingencyManager is produced when a :py:class:`BinaryContingencyManager` is
    transformed. It is also possible to create a BasicContingencyManager from event counts or
    a contingency table, although this not a common user requirement.

    A contingency table is built only from event counts, losing the connection
    to the actual event tables in their full dimensionality. The event count data is much
    smaller than the full event tables, particularly when considering very large data sets
    like Numerical Weather Prediction (NWP) data, which could be terabytes to petabytes in
    size.

    By contrast, A :py:class:`BinaryContingencyManager` retains the full event data, which provides
    some more flexbility but may reduce efficiency.
    """

    def __init__(self, counts: dict):
        """
        Compute any arrays required and store the resulting counts.
        """
        for key, arr in counts.items():
            counts[key] = arr.compute()

        self.counts = counts
        self._make_xr_table()

    def _make_xr_table(
        self,
    ):
        """
        From a dictionary of the skill score elements, produce an xarray
        structure which corresponds.
        """
        ctable_list = []
        entry_name = []
        for i, j in self.counts.items():
            entry_name.append(i)
            ctable_list.append(j)

        xr_table = xr.concat(ctable_list, dim="contingency")
        xr_table["contingency"] = entry_name

        self.xr_table = xr_table

    def __str__(self):
        heading = "Contingency Manager (xarray view of table):"
        table = self.xr_table
        tablerepr = repr(table)
        final = "\n".join([heading, tablerepr])
        return final

    def get_counts(self) -> dict:
        """

        Returns:
            dict: A dictionary of the contingency table counts. Values are xr.DataArray instances.
            The keys are:

            - tp_count for true positive count
            - tn_count for true negative count,
            - fp_count for false positive count
            - fn_count for false negative count
            - total_count for the total number of events.

        Here is an example of what may be returned:

        .. code-block :: python

            {'tp_count': <xarray.DataArray ()> Size: 8B array(5.),
             'tn_count': <xarray.DataArray ()> Size: 8B array(11.),
             'fp_count': <xarray.DataArray ()> Size: 8B array(1.),
             'fn_count': <xarray.DataArray ()> Size: 8B array(1.),
             'total_count': <xarray.DataArray ()> Size: 8B array(18.)}

        """
        return self.counts

    def get_table(self) -> xr.DataArray:
        """
        Returns:
            xr.DataArray: The contingency table as an xarray object. Contains
            a coordinate dimension called 'contingency'. Valid coordinates
            for this dimenstion are:

            - tp_count for true positive count
            - tn_count for true negative count
            - fp_count for false positive count
            - fn_count for false negative count
            - total_count for the total number of events.

        Here is an example of what may be returned:

        .. code-block :: python

            array([ 5., 11.,  1.,  1., 18.])

            Coordinates:

                contingency
                (contingency)
                <U11
                'tp_count' ... 'total_count'

            Indexes:

                contingency
                PandasIndex

            Attributes: (0)

        """
        return self.xr_table

    def accuracy(self) -> xr.DataArray:
        """
        Identical to :py:func:`fraction_correct`.

        Accuracy calculates the proportion of forecasts which are correct.

        Returns:
            xr.DataArray: A DataArray containing the accuracy score

        .. math::
            \\text{accuracy} = \\frac{\\text{true positives} + \\text{true negatives}}{\\text{total count}}

        Notes:
            - Range: 0 to 1, where 1 indicates a perfect score.
            - "True positives" is the same at "hits".
            - "False negatives" is the same as "misses".

        References:
            https://www.cawcr.gov.au/projects/verification/#ACC
        """
        count_dictionary = self.counts
        correct_count = count_dictionary["tp_count"] + count_dictionary["tn_count"]
        ratio = correct_count / count_dictionary["total_count"]
        return ratio

    def base_rate(self) -> xr.DataArray:
        """
        The observed event frequency.

        Returns:
            xr.DataArray: An xarray object containing the base rate.

        .. math::
            \\text{base rate} = \\frac{\\text{true positives} + \\text{false negatives}}{\\text{total count}}

        Notes:
            - Range: 0 to 1, where 1 indicates the event occurred every time.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".

        References:
            Hogan, R. J. & Mason, I. B. (2011). Deterministic forecasts of binary events.
            In I. T. Jolliffe & D. B. Stephenson (Eds.), Forecast verification: A practitioner's guide in atmospheric science (2nd ed.,
            pp. 39-51). https://doi.org/10.1002/9781119960003.ch3
        """
        cd = self.counts
        br = (cd["tp_count"] + cd["fn_count"]) / cd["total_count"]
        return br

    def forecast_rate(self) -> xr.DataArray:
        """
        The forecast event frequency.

        Returns:
            xr.DataArray: An xarray object containing the forecast rate.

        .. math::
            \\text{forecast rate} = \\frac{\\text{true positives} + \\text{false positives}}{\\text{total count}}

        Notes:
            - Range: 0 to 1, where 1 indicates the event was forecast every time.
            - "True positives" is the same as "hits".
            - "False positives" is the same as "false alarms".

        References:
            Hogan, R. J. & Mason, I. B. (2011). Deterministic forecasts of binary events.
            In I. T. Jolliffe & D. B. Stephenson (Eds.), Forecast verification: A practitioner's guide in atmospheric science (2nd ed.,
            pp. 39-51). https://doi.org/10.1002/9781119960003.ch3
        """
        cd = self.counts
        br = (cd["tp_count"] + cd["fp_count"]) / cd["total_count"]
        return br

    def fraction_correct(self) -> xr.DataArray:
        """
        Identical to :py:func:`accuracy`.

        Fraction correct calculates the proportion of forecasts which are correct.

        Returns:
            xr.DataArray: An xarray object containing the fraction correct.

        .. math::
            \\text{fraction correct} = \\frac{\\text{true positives} + \\text{true negatives}}{\\text{total count}}

        Notes:
            - Range: 0 to 1, where 1 indicates a perfect score.
            - "True positives" is the same as "hits".
            - "True negatives" is the same as "correct negatives".

        References:
            https://www.cawcr.gov.au/projects/verification/#ACC
        """
        return self.accuracy()

    def frequency_bias(self) -> xr.DataArray:
        """
        Identical to :py:func:`bias_score`.

        How did the forecast frequency of "yes" events compare to the observed frequency of "yes" events?

        Returns:
            xr.DataAray: An xarray object containing the frequency bias

        .. math::
            \\text{frequency bias} = \\frac{\\text{true positives} + \\text{false positives}}{\\text{true positives} + \\text{false negatives}}

        Notes:
            - Range: 0 to ∞ (infinity), where 1 indicates a perfect score.
            - "True positives" is the same as "hits".
            - "False positives" is the same as "false alarms".
            - "False negatives" is the same as "misses".

        References:
            https://www.cawcr.gov.au/projects/verification/#BIAS
        """
        # Note - bias_score calls this method
        cd = self.counts
        freq_bias = (cd["tp_count"] + cd["fp_count"]) / (cd["tp_count"] + cd["fn_count"])

        return freq_bias

    def bias_score(self) -> xr.DataArray:
        """
        Identical to :py:func:`frequency_bias`.

        How did the forecast frequency of "yes" events compare to the observed frequency of "yes" events?

        Returns:
            xr.DataArray: An xarray object containing the bias score

        .. math::
            \\text{frequency bias} = \\frac{\\text{true positives} + \\text{false positives}}{\\text{true positives} + \\text{false negatives}}

        Notes:
            - Range: 0 to ∞ (infinity), where 1 indicates a perfect score.
            - "True positives" is the same as "hits".
            - "False positives" is the same as "false alarms".
            - "False negatives" is the same as "misses".

        References:
            https://www.cawcr.gov.au/projects/verification/#BIAS
        """
        return self.frequency_bias()

    def hit_rate(self) -> xr.DataArray:
        """
        Identical to :py:func:`true_positive_rate`, :py:func:`probability_of_detection <BasicContingencyManager.probability_of_detection>`,
        :py:func:`sensitivity` and :py:func:`recall`.

        Calculates the proportion of the observed events that were correctly forecast.

        Returns:
            xr.DataArray: An xarray object containing the hit rate

        .. math::
            \\text{true positives} = \\frac{\\text{true positives}}{\\text{true positives} + \\text{false negatives}}

        Notes:
            - Range: 0 to 1.  Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".

        References:
            https://www.cawcr.gov.au/projects/verification/#POD
        """
        return self.probability_of_detection()

    def probability_of_detection(self) -> xr.DataArray:
        """
        Probability of detection (POD) is identical to :py:func:`hit_rate`, :py:func:`true_positive_rate`,
        :py:func:`sensitivity` and :py:func:`recall`.

        Calculates the proportion of the observed events that were correctly forecast.

        Returns:
            xr.DataArray: An xarray object containing the probability of detection

        .. math::
            \\text{hit rate} = \\frac{\\text{true positives}}{\\text{true positives} + \\text{false negatives}}

        Notes:
            - Range: 0 to 1.  Perfect score: 1.
            - "True positives" is the same as "hits"
            - "False negatives" is the same as "misses"

        References:
            https://www.cawcr.gov.au/projects/verification/#POD
        """
        # Note - hit_rate and sensitiviy call this function
        cd = self.counts
        pod = cd["tp_count"] / (cd["tp_count"] + cd["fn_count"])

        return pod

    def true_positive_rate(self) -> xr.DataArray:
        """
        Identical to :py:func:`hit_rate`, :py:func:`probability_of_detection <BasicContingencyManager.probability_of_detection>`,
        :py:func:`sensitivity` and :py:func:`recall`.

        The proportion of the observed events that were correctly forecast.

        Returns:
            xr.DataArray: An xarray object containing the true positive rate

        .. math::
            \\text{true positive rate} = \\frac{\\text{true positives}}{\\text{true positives} + \\text{false negatives}}

        Notes:
            - Range: 0 to 1.  Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".

        References:
            https://www.cawcr.gov.au/projects/verification/#POD
        """
        return self.probability_of_detection()

    def false_alarm_ratio(self) -> xr.DataArray:
        """
        The false alarm ratio (FAR) calculates the fraction of the predicted "yes" events
        which did not eventuate (i.e., were false alarms).

        Returns:
            xr.DataArray: An xarray object containing the false alarm ratio

        .. math::
            \\text{false alarm ratio} = \\frac{\\text{false positives}}{\\text{true positives} + \\text{false positives}}

        Notes:
            - Range: 0 to 1. Perfect score: 0.
            - Not to be confused with the False Alarm Rate.
            - "False positives" is the same as "false alarms".
            - "True positives" is the same as "hits".

        References:
            https://www.cawcr.gov.au/projects/verification/#FAR
        """
        cd = self.counts
        far = cd["fp_count"] / (cd["tp_count"] + cd["fp_count"])

        return far

    def false_alarm_rate(self) -> xr.DataArray:
        """
        Identical to :py:func:`probability_of_false_detection <BasicContingencyManager.probability_of_false_detection>`.

        What fraction of the non-events were incorrectly predicted?

        Returns:
            xr.DataArray: An xarray object containing the false alarm rate

        .. math::
            \\text{false alarm rate} = \\frac{\\text{false positives}}{\\text{true negatives} + \\text{false positives}}

        Notes:
            - Range: 0 to 1.  Perfect score: 0.
            - Not to be confused with the false alarm ratio.
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".

        References:
            https://www.cawcr.gov.au/projects/verification/#POFD
        """
        # Note - probability of false detection calls this function
        cd = self.counts
        far = cd["fp_count"] / (cd["tn_count"] + cd["fp_count"])

        return far

    def probability_of_false_detection(self) -> xr.DataArray:
        """
        Identical to :py:func:`false_alarm_rate`.

        What fraction of the non-events were incorrectly predicted?

        Returns:
            xr.DataArray: An xarray object containing the probability of false detection

        .. math::
            \\text{probability of false detection} = \\frac{\\text{false positives}}{\\text{true negatives} +
            \\text{false positives}}

        Notes:
            - Range: 0 to 1.  Perfect score: 0.
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".

        References:
            https://www.cawcr.gov.au/projects/verification/#POFD
        """

        return self.false_alarm_rate()

    def success_ratio(self) -> xr.DataArray:
        """
        Identical to :py:func:`precision`.

        What proportion of the forecast events actually eventuated?

        Returns:
            xr.DataArray: An xarray object containing the success ratio

        .. math::
            \\text{success ratio} = \\frac{\\text{true positives}}{\\text{true positives} +
            \\text{false positives}}

        Notes:
            - Range: 0 to 1.  Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False positives" is the same as "misses".

        References:
            https://www.cawcr.gov.au/projects/verification/#SR
        """
        cd = self.counts
        sr = cd["tp_count"] / (cd["tp_count"] + cd["fp_count"])

        return sr

    def threat_score(self) -> xr.DataArray:
        """
        Identical to :py:func:`critical_success_index`.

        Returns:
            xr.DataArray: An xarray object containing the threat score

        .. math::
            \\text{threat score} = \\frac{\\text{true positives}}{\\text{true positives} +
            \\text{false positives} + \\text{false negatives}}

        Notes:
            - Range: 0 to 1, 0 indicates no skill. Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".

        References:
            https://www.cawcr.gov.au/projects/verification/#CSI
        """
        # Note - critical success index just calls this method

        cd = self.counts
        ts = cd["tp_count"] / (cd["tp_count"] + cd["fp_count"] + cd["fn_count"])
        return ts

    def critical_success_index(self) -> xr.DataArray:
        """
        Identical to :py:func:`threat_score`.

        Returns:
            xr.DataArray: An xarray object containing the critical success index

        .. math::
            \\text{threat score} = \\frac{\\text{true positives}}{\\text{true positives} +
            \\text{false positives} + \\text{false negatives}}

        Notes:
            - Range: 0 to 1, 0 indicates no skill. Perfect score: 1.
            - Often known as CSI.
            - "True positives" is the same as "hits"
            - "False positives" is the same as "false alarms"
            - "True negatives" is the same as "correct negatives"

        References:
            https://www.cawcr.gov.au/projects/verification/#CSI
        """
        return self.threat_score()

    def peirce_skill_score(self) -> xr.DataArray:
        """
        Identical to :py:func:`hanssen_and_kuipers_discriminant` and :py:func:`true_skill_statistic`.

        How well did the forecast separate the "yes" events from the "no" events?

        Returns:
            xr.DataArray: An xarray object containing the Peirce Skill Score

        .. math::
            \\text{Peirce skill score} = \\frac{\\text{true positives}}{\\text{true positives} + 
            \\text{false negatives}} - \\frac{\\text{false positives}}{\\text{false positives} + 
            \\text{true negatives}}

        Notes:
            - Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".

        References:
            - https://www.cawcr.gov.au/projects/verification/#HK
            - Peirce, C.S., 1884. The numerical measure of the success of predictions. \
              Science, ns-4(93), pp.453-454. https://doi.org/10.1126/science.ns-4.93.453.b
        """
        cd = self.counts
        component_a = cd["tp_count"] / (cd["tp_count"] + cd["fn_count"])
        component_b = cd["fp_count"] / (cd["fp_count"] + cd["tn_count"])
        skill_score = component_a - component_b
        return skill_score

    def true_skill_statistic(self) -> xr.DataArray:
        """
        Identical to :py:func:`peirce_skill_score` and :py:func:`hanssen_and_kuipers_discriminant`.

        How well did the forecast separate the "yes" events from the "no" events?

        Returns:
            xr.DataArray: An xarray object containing the true skill statistic

        .. math::
            \\text{true skill statistic} = \\frac{\\text{true positives}}{\\text{true positives} +
            \\text{false negatives}} - \\frac{\\text{false positives}}{\\text{false positives} +
            \\text{true negatives}}

        Notes:
            - Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".

        References:
            https://www.cawcr.gov.au/projects/verification/#HK
        """
        return self.peirce_skill_score()

    def hanssen_and_kuipers_discriminant(self) -> xr.DataArray:
        """
        Identical to :py:func:`peirce_skill_score` and :py:func:`true_skill_statistic`.

        How well did the forecast separate the "yes" events from the "no" events?

        Returns:
            xr.DataArray: An xarray object containing Hanssen and Kuipers' Discriminant

        .. math::
            \\text{HK} = \\frac{\\text{true positives}}{\\text{true positives} +
            \\text{false negatives}} - \\frac{\\text{false positives}}{\\text{false positives} +
            \\text{true negatives}}

        where :math:`\\text{HK}` is Hansen and Kuipers Discriminant

        Notes:
            - Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".

        References:
            https://www.cawcr.gov.au/projects/verification/#HK
        """
        return self.peirce_skill_score()

    def sensitivity(self) -> xr.DataArray:
        """
        Identical to :py:func:`hit_rate`, :py:func:`probability_of_detection <BasicContingencyManager.probability_of_detection>`,
        :py:func:`true_positive_rate`, and :py:func:`recall`.

        Calculates the proportion of the observed events that were correctly forecast.

        Returns:
            xr.DataArray: An xarray object containing the probability of detection

        .. math::
            \\text{sensitivity} = \\frac{\\text{true positives}}{\\text{true positives} + \\text{false negatives}}

        Notes:
            - Range: 0 to 1.  Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".

        References:
            - https://www.cawcr.gov.au/projects/verification/#POD
            - https://en.wikipedia.org/wiki/Sensitivity_and_specificity

        """
        return self.probability_of_detection()

    def specificity(self) -> xr.DataArray:
        """
        Identical to :py:func:`true_negative_rate`.

        The probability that an observed non-event will be correctly predicted.

        Returns:
            xr.DataArray: An xarray object containing the true negative rate (specificity).

        .. math::
            \\text{specificity} = \\frac{\\text{true negatives}}{\\text{true negatives} + \\text{false positives}}

        Notes:
            - "True negatives" is the same as "correct negatives".
            - "False positives" is the same as "false alarms".

        Reference:
            https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        """
        cd = self.counts
        s = cd["tn_count"] / (cd["tn_count"] + cd["fp_count"])
        return s

    def true_negative_rate(self) -> xr.DataArray:
        """
        Identical to :py:func:`specificity`.

        The probability that an observed non-event will be correctly predicted.

        Returns:
            xr.DataArray: An xarray object containing the true negative rate.

        .. math::
            \\text{true negative rate} = \\frac{\\text{true negatives}}{\\text{true negatives} + \\text{false positives}}

        Notes:
            - "True negatives" is the same as "correct negatives".
            - "False positives" is the same as "false alarms".

        Reference:
            https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        """
        return self.specificity()

    def recall(self) -> xr.DataArray:
        """
        Identical to :py:func:`hit_rate`, :py:func:`probability_of_detection <BasicContingencyManager.probability_of_detection>`,
        :py:func:`true_positive_rate`, and :py:func:`sensitivity`.

        Calculates the proportion of the observed events that were correctly forecast.

        Returns:
            xr.DataArray: An xarray object containing the probability of detection

        .. math::
            \\text{recall} = \\frac{\\text{true positives}}{\\text{true positives} + \\text{false negatives}}

        Notes:
            - Range: 0 to 1.  Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".

        References:
            - https://www.cawcr.gov.au/projects/verification/#POD
            - https://en.wikipedia.org/wiki/Precision_and_recall
        """
        return self.probability_of_detection()

    def precision(self) -> xr.DataArray:
        """
        Identical to :py:func:`success_ratio`.

        What proportion of the forecast events actually eventuated?

        Returns:
            xr.DataArray: An xarray object containing the precision score

        .. math::
            \\text{precision} = \\frac{\\text{true positives}}{\\text{true positives} + \\text{false positives}}

        Notes:
            - Range: 0 to 1.  Perfect score: 1.
            - "True positives" is the same as "hits"
            - "False positives" is the same as "misses"

        References:
            - https://www.cawcr.gov.au/projects/verification/#SR
            - https://en.wikipedia.org/wiki/Precision_and_recall

        """
        return self.success_ratio()

    def f1_score(self) -> xr.DataArray:
        """
        Calculates the F1 score.

        Returns:
            xr.DataArray: An xarray object containing the F1 score

        .. math::
            \\text{F1} = \\frac{2 \\cdot \\text{true positives}}{(2 \\cdot  \\text{true positives}) +
            \\text{false positives} + \\text{false negatives}}

        Notes:
            - "True positives" is the same as "hits".
            - "False positives" is the same as "false alarms".
            - "False negatives" is the same as "misses".

        References:
            - https://en.wikipedia.org/wiki/F-score
        """
        cd = self.counts
        f1 = 2 * cd["tp_count"] / (2 * cd["tp_count"] + cd["fp_count"] + cd["fn_count"])
        return f1

    def equitable_threat_score(self) -> xr.DataArray:
        """
        Identical to :py:func:`gilberts_skill_score`.

        Calculates the Equitable threat score.

        How well did the forecast "yes" events correspond to the observed "yes"
        events (accounting for hits due to chance)?

        Returns:
            xr.DataArray: An xarray object containing the equitable threat score

        .. math::
            \\text{ETS} = \\frac{\\text{true positives} - \\text{true positives} _\\text{random}}\
            {\\text{true positives} + \\text{false negatives} + \\text{false positives} - 
            \\text{true positives} _\\text{random}}

        where

        .. math::
            \\text{true_positives}_{\\text{random}} = \\frac{(\\text{true positives} + 
            \\text{false negatives}) (\\text{true positives} + \\text{false positives})}{\\text{total count}}

        Notes:
            - Range: -1/3 to 1, 0 indicates no skill. Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".
            - "False positives" is the same as "false alarms"

        References:
            - Gilbert, G.K., 1884. Finley’s tornado predictions. American Meteorological Journal, 1(5), pp.166–172.
            - Hogan, R.J., Ferro, C.A., Jolliffe, I.T. and Stephenson, D.B., 2010. \
                Equitability revisited: Why the “equitable threat score” is not equitable. \
                Weather and Forecasting, 25(2), pp.710-726. https://doi.org/10.1175/2009WAF2222350.1
        """
        cd = self.counts
        hits_random = (cd["tp_count"] + cd["fn_count"]) * (cd["tp_count"] + cd["fp_count"]) / cd["total_count"]
        ets = (cd["tp_count"] - hits_random) / (cd["tp_count"] + cd["fn_count"] + cd["fp_count"] - hits_random)

        return ets

    def gilberts_skill_score(self) -> xr.DataArray:
        """
        Identical to :py:func:`equitable_threat_score`.

        Calculates the Gilbert skill score.

        How well did the forecast "yes" events correspond to the observed "yes"
        events (accounting for hits due to chance)?

        Returns:
            xr.DataArray: An xarray object containing the Gilberts Skill Score

        .. math::
            \\text{GSS} = \\frac{\\text{true positives} - \\text{true positives} _\\text{random}}\
            {\\text{true positives} + \\text{false negatives} + \\text{false positives} - 
            \\text{true positivies} _\\text{random}}

        where

        .. math::
            \\text{true_positives}_{\\text{random}} = \\frac{(\\text{true positives} + 
            \\text{false negatives}) (\\text{true positives} + \\text{false positives})}{\\text{total count}}

        Notes:
            - Range: -1/3 to 1, 0 indicates no skill. Perfect score: 1.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses"
            - "False positives" is the same as "false alarms".

        References:
            - Gilbert, G.K., 1884. Finley’s tornado predictions. American Meteorological Journal, 1(5), pp.166–172.
            - Hogan, R.J., Ferro, C.A., Jolliffe, I.T. and Stephenson, D.B., 2010. \
                Equitability revisited: Why the “equitable threat score” is not equitable. \
                Weather and Forecasting, 25(2), pp.710-726. https://doi.org/10.1175/2009WAF2222350.1
        """
        return self.equitable_threat_score()

    def heidke_skill_score(self) -> xr.DataArray:
        """
        Identical to :py:func:`cohens_kappa`.

        Calculates the Heidke skill score.

        What was the accuracy of the forecast relative to that of random chance?

        Returns:
            xr.DataArray: An xarray object containing the Heidke skill score

        .. math::
            \\\\
            \\text{HSS} =
                \\frac{\\text{true positives} + \\text{true negatives} - E_{random}}{
                       \\text{total count} - E_{\\text{random}}}

        where

        .. math::
            \\begin{aligned}
                E_{\\text{random}}
                    &= \\text{expected correct matches due to random chance}
                \\\\&= \\frac{\\left(\\text{tp} + \\text{fn}\\right) \\cdot
                              \\left(\\text{tp} + \\text{fp}\\right) +
                              \\left(\\text{tn} + \\text{fn}\\right) \\cdot
                              \\left(\\text{tn} + \\text{fp}\\right)}{
                              \\text{total count}}
            \\end{aligned}

        Notes:
            - Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
            - HSS = Heidke Skill Score
            - "True positives" (:math:`\\text{tp}`) is the same as "hits".
            - "False negatives" (:math:`\\text{fn}`) is the same as "misses".
            - "False positives" (:math:`\\text{fp}`) is the same as "false alarms".
            - "True negatives" (:math:`\\text{tn}`) is the same as "correct negatives".

        References:
            - https://en.wikipedia.org/wiki/Cohen%27s_kappa
        """
        cd = self.counts
        exp_correct = (1 / cd["total_count"]) * (
            (cd["tp_count"] + cd["fn_count"]) * (cd["tp_count"] + cd["fp_count"])
            + ((cd["tn_count"] + cd["fn_count"]) * (cd["tn_count"] + cd["fp_count"]))
        )
        hss = ((cd["tp_count"] + cd["tn_count"]) - exp_correct) / (cd["total_count"] - exp_correct)
        return hss

    def cohens_kappa(self) -> xr.DataArray:
        """
        Identical to :py:func:`heidke_skill_score`.

        Calculates Cohen's kappa.

        What was the accuracy of the forecast relative to that of random chance?

        Returns:
            xr.DataArray: An xarray object containing the Cohen's Kappa score

        .. math::
            \\\\
            \\text{Cohen's Kappa} \\left(\\kappa\\right) =
                \\frac{\\text{true positives} + \\text{true negatives} - E_{random}}{
                       \\text{total count} - E_{\\text{random}}}

        where

        .. math::
            \\begin{aligned}
                E_{\\text{random}}
                    &= \\text{expected correct matches due to random chance}
                \\\\&= \\frac{\\left(\\text{tp} + \\text{fn}\\right) \\cdot
                              \\left(\\text{tp} + \\text{fp}\\right) +
                              \\left(\\text{tn} + \\text{fn}\\right) \\cdot
                              \\left(\\text{tn} + \\text{fp}\\right)}{
                              \\text{total count}}
            \\end{aligned}

        Notes:
            - Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
            - "True positives" (:math:`\\text{tp}`) is the same as "hits".
            - "False negatives" (:math:`\\text{fn}`) is the same as "misses".
            - "False positives" (:math:`\\text{fp}`) is the same as "false alarms".
            - "True negatives" (:math:`\\text{tn}`) is the same as "correct negatives".

        References:
            - https://en.wikipedia.org/wiki/Cohen%27s_kappa
        """
        return self.heidke_skill_score()

    def odds_ratio(self) -> xr.DataArray:
        """
        Calculates the odds ratio

        What is the ratio of the odds of a "yes" forecast being correct, to the odds of
        a "yes" forecast being wrong?

        Returns:
            xr.DataArray: An xarray object containing the odds ratio

        .. math::
            \\begin{aligned}
                \\text{odds ratio} &=
                    \\left[\\frac{\\text{POD}}{\\text{1 - POD}}\\right]
                        \\div
                    \\left[\\frac{\\text{POFD}}{\\text{1 - POFD}}\\right]
                              \\\\ &=
                    \\frac{\\text{true positives} \\cdot \\text{true negatives}}{\\text{false positives} \\cdot \\text{false negatives}}
            \\end{aligned}

        where

        .. math::
            \\text{POD} = \\frac{\\text{true positives}}{\\text{true positives} + \\text{false negatives}}

        and

        .. math::
            \\text{POFD} = \\frac{\\text{false positives}}{\\text{true negatives} + \\text{false positives}}


        Notes:
            - Range: 0 to ∞, 1 indicates no skill. Perfect score: ∞.
            - POD = Probability of Detection
            - POFD = Probability of False Detection
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".

        References:
            - Stephenson, D.B., 2000. Use of the “odds ratio” for diagnosing forecast skill. \
              Weather and Forecasting, 15(2), pp.221-232. \
              https://doi.org/10.1175/1520-0434(2000)015%3C0221:UOTORF%3E2.0.CO;2
        """
        odds_r = (self.probability_of_detection() / (1 - self.probability_of_detection())) / (
            self.probability_of_false_detection() / (1 - self.probability_of_false_detection())
        )
        return odds_r

    def odds_ratio_skill_score(self) -> xr.DataArray:
        """
        Identical to :py:func:`yules_q`.

        Calculates the odds ratio skill score.

        What was the improvement of the forecast over random chance?

        Returns:
            xr.DataArray: An xarray object containing the odds ratio skill score

        .. math::

            \\begin{aligned}
                \\text{ORSS} &= \\frac{\\text{OR} - 1}{\\text{OR} + 1}
                        \\\\ &= \\frac{\\text{true positives} \\cdot \\text{true negatives}
                                         - \\text{false positives} \\cdot \\text{false negatives}}{
                                       \\text{true positives} \\cdot \\text{true negatives}
                                         + \\text{false positives} \\cdot \\text{false negatives}}
            \\end{aligned}

        Notes:
            - Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
            - ORSS = Odds Ratio Skill Score
            - OR = Odds ratio, see: :meth:`odds_ratio`
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".

        References:
            - Stephenson, D.B., 2000. Use of the “odds ratio” for diagnosing forecast skill. \
              Weather and Forecasting, 15(2), pp.221-232. \
              https://doi.org/10.1175/1520-0434(2000)015%3C0221:UOTORF%3E2.0.CO;2
        """
        cd = self.counts
        orss = (cd["tp_count"] * cd["tn_count"] - cd["fn_count"] * cd["fp_count"]) / (
            cd["tp_count"] * cd["tn_count"] + cd["fn_count"] * cd["fp_count"]
        )
        return orss

    def yules_q(self) -> xr.DataArray:
        """
        Identical to :py:func:`odds_ratio_skill_score`.

        Calculates the Yule's Q.

        What was the improvement of the forecast over random chance?

        Returns:
            xr.DataArray: An xarray object containing Yule's Q

        .. math::

            \\begin{aligned}
                \\text{Yule's Q} &= \\frac{\\text{OR} - 1}{\\text{OR} + 1}
                        \\\\ &= \\frac{\\text{true positives} \\cdot \\text{true negatives}
                                         - \\text{false positives} \\cdot \\text{false negatives}}{
                                       \\text{true positives} \\cdot \\text{true negatives}
                                         + \\text{false positives} \\cdot \\text{false negatives}}
            \\end{aligned}

        Notes:
            - Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
            - OR = Odds ratio, see: :meth:`odds_ratio`.
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".

        References:
            Stephenson, D.B., 2000. Use of the “odds ratio” for diagnosing forecast skill. \
            Weather and Forecasting, 15(2), pp.221-232. \
            https://doi.org/10.1175/1520-0434(2000)015%3C0221:UOTORF%3E2.0.CO;2
        """
        return self.odds_ratio_skill_score()

    def symmetric_extremal_dependence_index(self) -> xr.DataArray:
        """
        Calculates the Symmetric Extremal Dependence Index (SEDI).

        Returns:
            xr.DataArray: An xarray object containing the SEDI score

        .. math::
            \\frac{\\ln(\\text{POFD}) - \\ln(\\text{POD}) + \\ln(\\text{1-POD}) - \\ln(\\text{1 -POFD})}
            {\\ln(\\text{POFD}) + \\ln(\\text{POD}) + \\ln(\\text{1-POD}) + \\ln(\\text{1 -POFD})}

        where

        .. math::
            \\text{POD} = \\frac{\\text{true positives}}{\\text{true positives} + \\text{false negatives}}

        and

        .. math::
            \\text{POFD} = \\frac{\\text{false positives}}{\\text{true negatives} + \\text{false positives}}


        Notes:
            - POD = Probability of Detection
            - POFD = Probability of False Detection
            - "True positives" is the same as "hits".
            - "False negatives" is the same as "misses".
            - "False positives" is the same as "false alarms".
            - "True negatives" is the same as "correct negatives".
            - Range: -1 to 1, Perfect score: 1.


        References:
            Ferro, C.A.T. and Stephenson, D.B., 2011. Extremal dependence indices: Improved verification
            measures for deterministic forecasts of rare binary events. Weather and Forecasting, 26(5), pp.699-713.
            https://doi.org/10.1175/WAF-D-10-05030.1
        """
        score = (
            np.log(self.probability_of_false_detection())
            - np.log(self.probability_of_detection())
            + np.log(1 - self.probability_of_detection())
            - np.log(1 - self.probability_of_false_detection())
        ) / (
            np.log(self.probability_of_false_detection())
            + np.log(self.probability_of_detection())
            + np.log(1 - self.probability_of_detection())
            + np.log(1 - self.probability_of_false_detection())
        )
        return score


class BinaryContingencyManager(BasicContingencyManager):
    """
    See https://scores.readthedocs.io/en/stable/tutorials/Binary_Contingency_Scores.html for
    a detailed walkthrough showing the use in practice.

    A BinaryContingencyManager holds the underlying binary forecast and observed event data,
    from which it builds a contingency table and provides scoring functionality based on that table.

    A BinaryContingencyManager is typically created by an :py:class:`EventOperator`, such as a
    :py:class:`ThresholdOperator`, but can also be created directly from binary data if the user
    wishes.

    Supported operations include:
        - "Transforming" the data in various ways, such as dimensional reduction
        - Producing contingency tables
        - Calculating scores and metrics based on contingency tables

    The full data comprises several n-dimensional binary arrays which can be considered maps of:
        - True positives (hits)
        - False positives (false alarms)
        - True negatives (correct negatives)
        - False negatives (misses)

    These masks, in addition to supporting score calculations, can be accessed and used for:

    - Plotting these attributes on a map
    - Masking these values by a geographical region prior to score calculation

    As such, the per-pixel information is useful as well as the overall ratios involved.

    BinaryContingencyManager inherits from (uses) the :py:class:`BasicContingencyManager` class to
    provide score calculations on the final contingency table. Documentation for the available scores
    is found in the :py:class:`BasicContingencyManager` API entry but the calls can be made directly
    against instances of BinaryContingencyManager where performance or transformation are not a concern.
    """

    def __init__(
        self, fcst_events: FlexibleArrayType, obs_events: FlexibleArrayType
    ):  # pylint: disable=super-init-not-called
        self.fcst_events = fcst_events
        self.obs_events = obs_events

        self.tp = (self.fcst_events == 1) & (self.obs_events == 1)  # true positives
        self.tn = (self.fcst_events == 0) & (self.obs_events == 0)  # true negatives
        self.fp = (self.fcst_events == 1) & (self.obs_events == 0)  # false positives
        self.fn = (self.fcst_events == 0) & (self.obs_events == 1)  # false negatives

        # Bring back NaNs where there is either a forecast or observed event nan
        self.tp = self.tp.where(~np.isnan(fcst_events))
        self.tp = self.tp.where(~np.isnan(obs_events))
        self.tn = self.tn.where(~np.isnan(fcst_events))
        self.tn = self.tn.where(~np.isnan(obs_events))
        self.fp = self.fp.where(~np.isnan(fcst_events))
        self.fp = self.fp.where(~np.isnan(obs_events))
        self.fn = self.fn.where(~np.isnan(fcst_events))
        self.fn = self.fn.where(~np.isnan(obs_events))

        # Variables for count-based metrics
        self.counts = self._get_counts()
        self._make_xr_table()

    def transform(
        self,
        *,
        reduce_dims: Optional[FlexibleDimensionTypes] = None,
        preserve_dims: Optional[FlexibleDimensionTypes] = None,
    ) -> BasicContingencyManager:
        """
        Compute the contingency table, preserving or reducing the specified dimensions.

        Args:
            - reduce_dims: Dimensions to reduce. Can be "all" to reduce all dimensions.
            - preserve_dims: Dimensions to preserve. Can be "all" to preserve all dimensions.

        Returns:
            scores.categorical.BasicContingencyManager: A `scores` class which supports efficient
            calculation of contingency metrics.
        """
        cd = self._get_counts(reduce_dims=reduce_dims, preserve_dims=preserve_dims)
        return BasicContingencyManager(cd)

    def _get_counts(
        self,
        *,
        reduce_dims: Optional[FlexibleDimensionTypes] = None,
        preserve_dims: Optional[FlexibleDimensionTypes] = None,
    ) -> dict:
        """
        Generates the uncomputed count values
        """

        to_reduce = scores.utils.gather_dimensions(
            self.fcst_events.dims,
            self.obs_events.dims,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        cd = {
            "tp_count": self.tp.sum(dim=to_reduce),
            "tn_count": self.tn.sum(dim=to_reduce),
            "fp_count": self.fp.sum(dim=to_reduce),
            "fn_count": self.fn.sum(dim=to_reduce),
        }
        total = cd["tp_count"] + cd["tn_count"] + cd["fp_count"] + cd["fn_count"]
        cd["total_count"] = total

        return cd


class EventOperator(ABC):
    """
    Abstract Base Class (ABC) for event operators which can be used in deriving contingency
    tables. ABCs are not used directly but instead define the requirements for other classes
    and may be used for type checking.
    """

    @abstractmethod
    def make_event_tables(
        self,
        fcst: FlexibleArrayType,
        obs: FlexibleArrayType,
        *,
        event_threshold=None,
        op_fn=None,
    ):
        """
        This method should be over-ridden to return forecast and observed event tables
        """
        ...  # pragma: no cover # pylint: disable=unnecessary-ellipsis

    @abstractmethod
    def make_contingency_manager(
        self,
        fcst: FlexibleArrayType,
        obs: FlexibleArrayType,
        *,
        event_threshold=None,
        op_fn=None,
    ):
        """
        This method should be over-ridden to return a contingency table.
        """
        ...  # pragma: no cover # pylint: disable=unnecessary-ellipsis


class ThresholdEventOperator(EventOperator):
    """
    See https://scores.readthedocs.io/en/stable/tutorials/Binary_Contingency_Scores.html for
    a detailed walkthrough showing the use in practice.

    A ThresholdEventOperator is used to produce a :py:class:`BinaryContingencyManager` from
    forecast and observed data. It considers an event to be defined when the forecast and
    observed variables meet a particular threshold condition (e.g. rainfall above 1mm).

    The class may be used for any variable, for any threshold, and for any comparison
    operator (e.g. greater-than, less-than, greater-than-or-equal-to, ... )
    """

    def __init__(
        self,
        *,
        precision=DEFAULT_PRECISION,
        default_event_threshold=0.001,
        default_op_fn=operator.ge,
    ):
        self.precision = precision
        self.default_event_threshold = default_event_threshold
        self.default_op_fn = default_op_fn

    def make_event_tables(
        self,
        fcst: FlexibleArrayType,
        obs: FlexibleArrayType,
        *,
        event_threshold=None,
        op_fn=None,
    ):
        """
        Using this function requires a careful understanding of the structure of the data
        and the use of the operator function. The default operator is a simple greater-than
        operator, so this will work on a simple DataArray. To work on a DataSet, a richer
        understanding is required. It is recommended to work through the tutorial at
        https://scores.readthedocs.io/en/stable/tutorials/Binary_Contingency_Scores.html .
        This tutorial reviews more complex use cases, including multivariate gridded model
        data, and station data structures.
        """

        if not event_threshold:
            event_threshold = self.default_event_threshold

        if op_fn is None:
            op_fn = self.default_op_fn

        fcst_events = op_fn(fcst, event_threshold)
        obs_events = op_fn(obs, event_threshold)

        # Bring back NaNs
        fcst_events = fcst_events.where(~np.isnan(fcst))
        obs_events = obs_events.where(~np.isnan(obs))

        return (fcst_events, obs_events)

    def make_contingency_manager(
        self,
        fcst: FlexibleArrayType,
        obs: FlexibleArrayType,
        *,
        event_threshold=None,
        op_fn=None,
    ) -> BinaryContingencyManager:
        """
        Using this function requires a careful understanding of the structure of the data
        and the use of the operator function. The default operator is a simple greater-than
        operator, so this will work on a simple DataArray. To work on a DataSet, a richer
        understanding is required. It is recommended to work through the tutorial at
        https://scores.readthedocs.io/en/stable/tutorials/Binary_Contingency_Scores.html .
        This tutorial reviews more complex use cases, including multivariate gridded model
        data, and station data structures.
        """

        if not event_threshold:
            event_threshold = self.default_event_threshold

        if op_fn is None:
            op_fn = self.default_op_fn

        fcst_events = op_fn(fcst, event_threshold)
        obs_events = op_fn(obs, event_threshold)

        # Bring back NaNs
        fcst_events = fcst_events.where(~np.isnan(fcst))
        obs_events = obs_events.where(~np.isnan(obs))

        table = BinaryContingencyManager(fcst_events, obs_events)
        return table
