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
from abc import ABC, abstractmethod
from typing import Optional
import xarray as xr

import scores.utils
from scores.typing import FlexibleArrayType, FlexibleDimensionTypes

DEFAULT_PRECISION = 8


class BasicContingencyTable:
    """
    A BasicContingencyTable is produced when a BinaryContingencyTable is transformed.

    A basic contingency table is built only from the event counts, losing the connection
    to the actual event tables in their full dimensionality.

    The event count data is much smaller than the full event tables, particularly when
    considering very large data sets like NWP data, which could be terabytes to petabytes
    in size.
    """

    def __init__(self, counts: dict):
        '''
        Compute any arrays required and store the resulting counts.
        '''
        for key, arr in counts.items():
            counts[key] = arr.compute()

        self.counts = counts
        self._make_xr_table()

    def _make_xr_table(self,):
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
        xr_table

        self.xr_table = xr_table

    def __repr__(self):
        heading = "Contingency Table (xarray view):"
        table = self.xr_table
        tablerepr = repr(table)
        final = '\n'.join([heading, tablerepr])
        return final

    def __str__(self):
        table = self.xr_table
        return str(table)

    def get_counts(self):
        return self.counts

    def get_table(self):
        return self.xr_table

    def accuracy(self):
        """
        The proportion of forecasts which are true
        """
        count_dictionary = self.counts
        correct_count = count_dictionary["tp_count"] + count_dictionary["tn_count"]
        ratio = correct_count / count_dictionary["total_count"]
        return ratio

    def frequency_bias(self):
        """
        How did the forecast frequency of "yes" events compare to the observed frequency of "yes" events?
        """
        cd = self.counts
        freq_bias = (cd["tp_count"] + cd["fp_count"]) / (cd["tp_count"] + cd["fn_count"])

        return freq_bias

    def hit_rate(self):
        """
        What proportion of the observed events where correctly forecast?
        Identical to probability_of_detection
        Range: 0 to 1.  Perfect score: 1.
        """
        cd = self.counts
        pod = cd["tp_count"] / (cd["tp_count"] + cd["fn_count"])

        return pod

    def probability_of_detection(self):
        """
        What proportion of the observed events where correctly forecast?
        Identical to hit_rate
        Range: 0 to 1.  Perfect score: 1.
        """
        cd = self.counts
        pod = cd["tp_count"] / (cd["tp_count"] + cd["fn_count"])

        return pod

    def false_alarm_rate(self):
        """
        What fraction of the non-events were incorrectly predicted?
        Identical to probability_of_false_detection
        Range: 0 to 1.  Perfect score: 0.
        """
        cd = self.counts
        far = cd["fp_count"] / (cd["tn_count"] + cd["fp_count"])

        return far

    def probability_of_false_detection(self):
        """
        What fraction of the non-events were incorrectly predicted?
        Identical to false_alarm_rate
        Range: 0 to 1.  Perfect score: 0.
        """
        cd = self.counts
        far = cd["fp_count"] / (cd["tn_count"] + cd["fp_count"])

        return far

    def success_ratio(self):
        """
        What proportion of the forecast events actually eventuated?
        Range: 0 to 1.  Perfect score: 1.
        """
        cd = self.counts
        sr = cd["tp_count"] / (cd["tp_count"] + cd["fp_count"])

        return sr

    def threat_score(self):
        """
        How well did the forecast "yes" events correspond to the observed "yes" events?
        Identical to critical_success_index
        Range: 0 to 1, 0 indicates no skill. Perfect score: 1.
        """
        cd = self.counts
        ts = cd["tp_count"] / (cd["tp_count"] + cd["fp_count"] + cd["tn_count"])
        return ts

    def critical_success_index(self):
        """
        How well did the forecast "yes" events correspond to the observed "yes" events?
        Identical to threat_score
        Range: 0 to 1, 0 indicates no skill. Perfect score: 1.
        """
        cd = self.counts
        ts = cd["tp_count"] / (cd["tp_count"] + cd["fp_count"] + cd["tn_count"])
        return ts

    def pierce_skill_score(self):
        """
        Hanssen and Kuipers discriminant (true skill statistic, Peirce's skill score)
        How well did the forecast separate the "yes" events from the "no" events?
        Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
        """
        cd = self.counts
        component_a = cd["tp_count"] / (cd["tp_count"] + cd["fn_count"])
        component_b = cd["fn_count"] / (cd["fn_count"] + cd["tn_count"])
        skill_score = component_a - component_b
        return skill_score

    def sensitivity(self):
        """
        https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        """
        cd = self.counts
        s = cd["tp_count"] / (cd["tp_count"] + cd["fn_count"])
        return s

    def specificity(self):
        """
        https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        """
        cd = self.counts
        s = cd["tn_count"] / (cd["tn_count"] + cd["fp_count"])
        return s


class BinaryContingencyTable(BasicContingencyTable):
    """
    At each location, the value will either be:
     - A true positive
     - A false positive
     - A true negative
     - A false negative

    It will be common to want to operate on masks of these values,
    such as:
     - Plotting these attributes on a map
     - Calculating the total number of these attributes
     - Calculating various ratios of these attributes, potentially
       masked by geographical area (e.g. accuracy in a region)

    As such, the per-pixel information is useful as well as the overall
    ratios involved.
    """

    def __init__(self, forecast_events: FlexibleArrayType, observed_events: FlexibleArrayType):
        # type checking goes here
        self.forecast_events = forecast_events
        self.observed_events = observed_events

        self.tp = (self.forecast_events == 1) & (self.observed_events == 1)  # true positives
        self.tn = (self.forecast_events == 0) & (self.observed_events == 0)  # true negatives
        self.fp = (self.forecast_events == 1) & (self.observed_events == 0)  # false positives
        self.fn = (self.forecast_events == 0) & (self.observed_events == 1)  # false negatives

        # Variables for count-based metrics
        self.counts = self._get_counts()
        self._make_xr_table()

    def transform(
        self,
        *,
        reduce_dims: Optional[FlexibleDimensionTypes] = None,
        preserve_dims: Optional[FlexibleDimensionTypes] = None,
    ):
        """
        Calculate and compute the contingency table according to the specified dimensions
        """
        cd = self._get_counts(reduce_dims=reduce_dims, preserve_dims=preserve_dims)
        return BasicContingencyTable(cd)

    def _get_counts(
        self,
        *,
        reduce_dims: Optional[FlexibleDimensionTypes] = None,
        preserve_dims: Optional[FlexibleDimensionTypes] = None,
    ):
        """
        Generates the uncomputed count values
        """

        to_reduce = scores.utils.gather_dimensions(
            self.forecast_events.dims,
            self.observed_events.dims,
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
    Base class for event operators which can be used in deriving contingency
    tables. This will be expanded as additional use cases are incorporated
    beyond the ThresholdEventOperator.
    """

    @abstractmethod
    def make_event_tables(
        self, forecast: FlexibleArrayType, observed: FlexibleArrayType, *, event_threshold=None, op_fn=operator.gt
    ):
        """
        This method should be over-ridden to return forecast and observed event tables
        """
        ...

    @abstractmethod
    def make_table(
        self, forecast: FlexibleArrayType, observed: FlexibleArrayType, *, event_threshold=None, op_fn=operator.gt
    ):
        """
        This method should be over-ridden to return a contingency table.
        """
        ...


class ThresholdEventOperator(EventOperator):
    """
    Given a forecast and and an observation, consider an event defined by
    particular variables meeting a threshold condition (e.g. rainfall above 1mm).

    This class abstracts that concept for any event definition.
    """

    def __init__(self, *, precision=DEFAULT_PRECISION, default_event_threshold=0.001):
        self.precision = precision
        self.default_event_threshold = default_event_threshold

    def make_event_tables(
        self, forecast: FlexibleArrayType, observed: FlexibleArrayType, *, event_threshold=None, op_fn=operator.gt
    ):
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

        return (forecast_events, observed_events)

    def make_table(
        self, forecast: FlexibleArrayType, observed: FlexibleArrayType, *, event_threshold=None, op_fn=operator.gt
    ):
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
