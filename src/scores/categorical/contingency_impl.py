'''
This foundation class provides an underpinning data structure which can be used for 
contingency tables of various kinds, also known as a confusion matrix.

It allows the careful setting out of when and where forecasts closely match
observations.

The simplest kind of table is an accuracy table, which is based on when the 
forecast and the obs match exactly or not. As real-world data is real-varying, this will
typically have been calculated as a binary flag which indicates when the forecast
and observed values are *sufficiently close* based on some tolerance.

The second-simplest kind of table is a binary contingency table which captures true
positives, true negatives, false positives and false negatives. 

A somewhat more complex type of table is a categorical contingency table, which includes
multiple categories of match. An example set of categoried might be "exact match",
"close match", "rough match" and "not a match". The category here is the category of match.

A point of possible confusion is that forecast and observations may be forecasts of
categories (often called 'events'). For example, the forecast might be "rain" and the
observation might be "no rain". The contingency table would then contain information
about when the forecast category matches the observed category.

The process of deriving a contingency table then relies on the forecast data, the observation
data, and a matching operator. The matching operator will produce the category from the
forecast and the observed data. Specific use cases may well be supported by a case-specific
matching operator.

Scores supports complex, weighted, multi-dimensional data, including in contingency tables.

The way to achieve this is by leveraging the scores themselves to apply the weighting step.
As such, all uses of the contingency tables should be for comparing the results of scoring
operatings which have themselves properly handled all dimensionality reduction and weighting.

A further complication resides in the information representation used to depict the categories.
There are multiple common schemes which are likely to be encountered by users. 

A simple example is where Python boolean True and False are used to indicate a binary match.
An alternative could be to use the integers 0 or 1.

A more complex example is that of categories. Category designations could equally be a series of 
numbers (e.g. 0, 1, 2, 3); "buckets" or ranges (e.g. 0-.9, 1-1.9, 2-2.9), or labels ("rain", "no rain")
or yet more complex examples and data types. Some practical decisions have to be made to provide a
useful implementation which is computationally efficient while being sufficiently flexible to 
support the expected requirements. The developers acknowledge some of these choices may not
match all user preferences, and would happily receive any feedback about alternative approaches.

The implementation in this module uses "True" and "False" for the binary table, and uses integers
for categorical data. Users wishing to match ranges or named categories should maintain a separate
data structure mapping between the integer values and the semantics of the category. For convenience,
all tables have a 'translation' dictionary which can be used to reference and dereference the categories.

Users can supply their own matching operators to the top-level module functions so long as
they can translate from a score (or proximity function) to either a BinaryTable or a 
CategoryTable.
'''

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
	'''
	Functional API example for contingency scores

	If the event operator is None, then fcst and obs are held to contain
	binary information about whether an event is predicted or occurred.
	If the event operator is provided, it will be utilised to produce the
	event tables for the forecast and observed conditions prior to generating
	the contingency table.
	'''

	if event_operator is not None:
		table = event_operator.make_table(fcst, obs)
	else:
		table = BinaryContingencyTable(fcst, obs)


	acc = table.accuracy(reduce_dims=reduce_dims, preserve_dims=preserve_dims)
	return acc




def make_binary_table(proximity, match_operator):
	table = match_operator.make_table(proximity)
	return table

def make_category_table(proximity, category_operator):
	table = category_operator.make_table(proximity)
	return table

class BinaryContingencyTable():
	'''
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
	'''

	def __init__(self, forecast_events, observed_events):

		# type checking goes here
		self.forecast_events = forecast_events
		self.observed_events = observed_events

		self.tp = (self.forecast_events == 1) & (self.observed_events == 1)  # true positives
		self.tn = (self.forecast_events == 0) & (self.observed_events == 0)  # true negatives
		self.fp = (self.forecast_events == 1) & (self.observed_events == 0)  # false positives
		self.fn = (self.forecast_events == 0) & (self.observed_events == 1)  # false negatives

		# Variables for count-based metrics, calculated on first access
		self.tp_count = self.tp.sum()     # Count of true positives
		self.tn_count = self.tn.sum()     # Count of true negatives
		self.fp_count = self.fp.sum()     # Count of false positives
		self.fn_count = self.fn.sum()     # Count of true negatives
		self.total_count = self.tp_count + self.tn_count + self.fp_count + self.fn_count

	def generate_counts(self, *, reduce_dims=None, preserve_dims=None):

		to_reduce = scores.utils.gather_dimensions(self.forecast_events.dims, self.observed_events.dims, 
			reduce_dims=reduce_dims, preserve_dims=preserve_dims)

		cd = {
			'tp_count': self.tp.sum(to_reduce),
			'tn_count': self.tn.sum(to_reduce),
			'fp_count': self.fp.sum(to_reduce),
			'fn_count': self.fn.sum(to_reduce),
		}
		total = cd['tp_count'] + cd['tn_count'] + cd['fp_count'] + cd['fn_count']
		cd['total_count'] = total

		return cd


	def accuracy(self, *, preserve_dims = None, reduce_dims=None):
		'''
		The proportion of forecasts which are true
		'''
		count_dictionary = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
		correct_count = count_dictionary['tp_count'] + count_dictionary['tn_count']
		ratio = correct_count / count_dictionary['total_count']
		return ratio


	def frequency_bias(self, *, preserve_dims = None, reduce_dims=None):
		'''
		'''
		cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
		freq_bias = (cd['tp_count'] + cd['fp_count']) / (cd['tp_count'] + cd['fn_count'])

		return freq_bias

	def probability_of_detection(self, *, preserve_dims = None, reduce_dims=None):
		'''
		What proportion of the observed events where correctly forecast?
		Range: 0 to 1.  Perfect score: 1.
		'''
		cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
		pod = cd['tp_count'] / (cd['tp_count'] + cd['fn_count'])

		return pod

	def false_alarm_rate(self, *, preserve_dims = None, reduce_dims=None):
		'''
		What fraction of the non-events were incorrectly predicted?
		Range: 0 to 1.  Perfect score: 0.
		'''
		cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
		far = cd['fp_count'] / (cd['tn_count'] + cd['fp_count'])

		return far

	def success_ratio(self, *, preserve_dims = None, reduce_dims=None):
		'''
		What proportion of the forecast events actually eventuated?
		Range: 0 to 1.  Perfect score: 1.
		'''
		cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
		sr = cd['tp_count'] / (cd['tp_count'] + cd['fp_count'])

		return sr

	def threat_score(self, *, preserve_dims = None, reduce_dims=None):
		'''
		How well did the forecast "yes" events correspond to the observed "yes" events?
		Range: 0 to 1, 0 indicates no skill. Perfect score: 1.
		'''
		cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
		ts = cd['tp_count'] / (cd['tp_count'] + cd['fp_count'] + cd['tn_count'])
		return ts

	def sensitivity(self, *, preserve_dims = None, reduce_dims=None):
		'''
		What proportion of non-events were correctly predicted?
		'''
		cd = self.generate_counts(preserve_dims=preserve_dims, reduce_dims=reduce_dims)
		s = cd['tn_count'] / (cd['tn_count'] + cd['fp_count'])
		return s

class MatchingOperator:
	pass

class EventThresholdOperator(MatchingOperator):
	'''
	Given a forecast and and an observation, consider an event defined by
	particular variables meeting a threshold condition (e.g. rainfall above 1mm).

	This class abstracts that concept for any event definition.
	'''

	def __init__(self, *, precision=DEFAULT_PRECISION, default_event_threshold=0.001):
		self.precision = precision
		self.default_event_threshold = default_event_threshold

	def make_table(self, forecast, observed, *, event_threshold=None, op_fn=operator.gt):
		'''
		Using this function requires a careful understanding of the structure of the data
		and the use of the operator function. The default operator is a simple greater-than
		operator, so this will work on a simple DataArray. To work on a DataSet, a richer
		understanding is required. It is recommended to work through the Contingency Table
		tutorial to review more complex use cases, including on multivariate gridded model
		data, and on station data structures.
		'''

		if not event_threshold:
			event_threshold = self.default_event_threshold

		forecast_events = op_fn(forecast, event_threshold)
		observed_events = op_fn(observed, event_threshold)

		table = BinaryContingencyTable(forecast_events, observed_events)
		return table

