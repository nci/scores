"""
This foundation class provides an underpinning data structure which can be used for 
contingency tables of various kinds, also known as a confusion matrix.

It allows the careful setting out of when and where forecasts closely match
observations.

The binary contingency table captures true positives, true negatives, false positives and 
false negatives. 

The process of deriving a contingency table relies on the forecast data, the observation
data, and a matching or event operator. The event operator will produce the category from the
forecast and the observed data. Specific use cases may well be supported by a case-specific
matching operator.

Scores supports complex, weighted, multi-dimensional data, including in contingency tables.

Users can supply their own event operators to the top-level module functions.
"""

import operator

import xarray as xr

import scores.utils
from scores.typing import FlexibleArrayType, FlexibleDimensionTypes

DEFAULT_PRECISION = 8


def accuracy(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    *,
    event_operator: None,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    weights: xr.DataArray = None,
):
    """
    Functional API example for contingency scores

    If the event operator is None, then fcst and obs are held to contain
    binary information about whether an event is predicted or occurred.
    If the event operator is provided, it will be utilised to produce the
    event tables for the forecast and observed conditions prior to generating
    the contingency table.
    """

    if event_operator is not None:
        table = event_operator.make_table(fcst, obs)
    else:
        table = BinaryContingencyTable(fcst, obs)

    acc = table.accuracy(reduce_dims=reduce_dims, preserve_dims=preserve_dims)
    return acc


class BinaryContingencyTable:
    """
    At each location, the value will either be:
     - A true positive    (0)
     - A false positive   (1)
     - A true negative    (2)
     - A false negative   (3)

    It will be common to want to operate on masks of these values,
    such as:
     - Plotting these attributes on a map
     - Calculating the total number of these attributes
     - Calculating various ratios of these attributes, potentially
       masked by geographical area (e.g. accuracy in a region)

    As such, the per-pixel information is useful as well as the overall
    ratios involved.
    """

    def __init__(self, forecast_events, observed_events):
        # type checking goes here
        self.forecast_events = forecast_events
        self.observed_events = observed_events

        self.tp = (self.forecast_events == 1) & (self.observed_events == 1)  # true positives
        self.tn = (self.forecast_events == 0) & (self.observed_events == 0)  # true negatives
        self.fp = (self.forecast_events == 1) & (self.observed_events == 0)  # false positives
        self.fn = (self.forecast_events == 0) & (self.observed_events == 1)  # false negatives

        # Variables for count-based metrics, calculated on first access
        self.tp_count = self.tp.sum()  # Count of true positives
        self.tn_count = self.tn.sum()  # Count of true negatives
        self.fp_count = self.fp.sum()  # Count of false positives
        self.fn_count = self.fn.sum()  # Count of true negatives
        self.total_count = self.tp_count + self.tn_count + self.fp_count + self.fn_count

    def generate_counts(self, *, reduce_dims=None, preserve_dims=None):
        to_reduce = scores.utils.gather_dimensions(
            self.forecast_events.dims, self.observed_events.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
        )

        cd = {
            "tp_count": self.tp.sum(to_reduce),
            "tn_count": self.tn.sum(to_reduce),
            "fp_count": self.fp.sum(to_reduce),
            "fn_count": self.fn.sum(to_reduce),
        }
        total = cd["tp_count"] + cd["tn_count"] + cd["fp_count"] + cd["fn_count"]
        cd["total_count"] = total

        return cd

    def accuracy(self, *, preserve_dims=None, reduce_dims=None):
        """
        The proportion of forecasts which are true
        """
        count_dictionary = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        correct_count = count_dictionary["tp_count"] + count_dictionary["tn_count"]
        ratio = correct_count / count_dictionary["total_count"]
        return ratio

    def frequency_bias(self, *, preserve_dims=None, reduce_dims=None):
        """ """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        freq_bias = (cd["tp_count"] + cd["fp_count"]) / (cd["tp_count"] + cd["fn_count"])

        return freq_bias

    def hit_rate(self, *, preserve_dims=None, reduce_dims=None):
        """
        What proportion of the observed events where correctly forecast?
        Identical to probability_of_detection
        Range: 0 to 1.  Perfect score: 1.
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        pod = cd["tp_count"] / (cd["tp_count"] + cd["fn_count"])

        return pod

    def probability_of_detection(self, *, preserve_dims=None, reduce_dims=None):
        """
        What proportion of the observed events where correctly forecast?
        Identical to hit_rate
        Range: 0 to 1.  Perfect score: 1.
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        pod = cd["tp_count"] / (cd["tp_count"] + cd["fn_count"])

        return pod

    def false_alarm_rate(self, *, preserve_dims=None, reduce_dims=None):
        """
        What fraction of the non-events were incorrectly predicted?
        Identical to probability_of_false_detection
        Range: 0 to 1.  Perfect score: 0.
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        far = cd["fp_count"] / (cd["tn_count"] + cd["fp_count"])

        return far

    def probability_of_false_detection(self, *, preserve_dims=None, reduce_dims=None):
        """
        What fraction of the non-events were incorrectly predicted?
        Identical to false_alarm_rate
        Range: 0 to 1.  Perfect score: 0.
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        far = cd["fp_count"] / (cd["tn_count"] + cd["fp_count"])

        return far

    def success_ratio(self, *, preserve_dims=None, reduce_dims=None):
        """
        What proportion of the forecast events actually eventuated?
        Range: 0 to 1.  Perfect score: 1.
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        sr = cd["tp_count"] / (cd["tp_count"] + cd["fp_count"])

        return sr

    def threat_score(self, *, preserve_dims=None, reduce_dims=None):
        """
        How well did the forecast "yes" events correspond to the observed "yes" events?
        Identical to critical_success_index
        Range: 0 to 1, 0 indicates no skill. Perfect score: 1.
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        ts = cd["tp_count"] / (cd["tp_count"] + cd["fp_count"] + cd["tn_count"])
        return ts

    def critical_success_index(self, *, preserve_dims=None, reduce_dims=None):
        """
        How well did the forecast "yes" events correspond to the observed "yes" events?
        Identical to threat_score
        Range: 0 to 1, 0 indicates no skill. Perfect score: 1.
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        ts = cd["tp_count"] / (cd["tp_count"] + cd["fp_count"] + cd["tn_count"])
        return ts        

    def pierce_skill_score(self, *, preserve_dims=preserve_dims, reduce_dims=reduce_dims):
        """
        Hanssen and Kuipers discriminant (true skill statistic, Peirce's skill score)
        How well did the forecast separate the "yes" events from the "no" events?
        Range: -1 to 1, 0 indicates no skill. Perfect score: 1.        
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        componentA = cd['tp_count'] / (cd['tp_count'] + cd['fn_count'])
        componentB = cd['fn_count'] / (cd['fn_count'] + cd['tn_count'])
        skill_score = componentA - componentB
        return skill_score

    def sensitivity(self, *, preserve_dims=None, reduce_dims=None):
        """        
        https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        s = cd["tp_count"] / (cd["tp_count"] + cd["fn_count"])
        return s

    def specificity(self, *, preserve_dims=None, reduce_dims=None):
        """        
        https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        """
        cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        s = cd["tn_count"] / (cd["tn_count"] + cd["fp_count"])
        return s        


class MatchingOperator:
    pass


class EventThresholdOperator(MatchingOperator):
    """
    Given a forecast and and an observation, consider an event defined by
    particular variables meeting a threshold condition (e.g. rainfall above 1mm).

    This class abstracts that concept for any event definition.
    """

    def __init__(self, *, precision=DEFAULT_PRECISION, default_event_threshold=0.001):
        self.precision = precision
        self.default_event_threshold = default_event_threshold

    def make_table(self, forecast, observed, *, event_threshold=None, op_fn=operator.gt):
        """
        Using this function requires a careful understanding of the structure of the data
        and the use of the operator function. The default operator is a simple greater-than
        operator, so this will work on a simple DataArray. To work on a DataSet, a richer
        understanding is required. It is recommended to work through the Contingency Table
        tutorial to review more complex use cases, including on multivariate gridded model
        data, and on station data structures.
        """

        if not event_threshold:
            event_threshold = self.default_event_threshold

        forecast_events = op_fn(forecast, event_threshold)
        observed_events = op_fn(observed, event_threshold)

        table = BinaryContingencyTable(forecast_events, observed_events)
        return table
