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
import xarray

DEFAULT_PRECISION = 8

def make_binary_table(proximity, match_operator):
	table = match_operator.make_table(proximity)
	return table

def make_category_table(proximity, category_operator):
	table = category_operator.make_table(proximity)
	return table

class ContingencyTable(xarray.DataArray):
	pass

class CategoryTable(xarray.DataArray):
	pass

class BinaryContingencyTable(xarray.DataArray):
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

@xarray.register_dataset_accessor("BinaryTable")
class BinaryTable:	
	'''
	At each location, the value will either be:
	 - True (Accurate)
	 - False (Not Accurate)

	It will be common to want to operate on masks of these values,
	such as:
	 - Plotting these attributes on a map
	 - Calculating the total number of these attributes
	 - Calculating various ratios of these attributes, potentially
	   masked by geographical area (e.g. accuracy in a region)

	As such, the per-pixel information is useful as well as the overall
	ratios involved.	
	'''

	def __init__(self, data_array):

		bool_array = data_array.astype(bool)
		self._obj = bool_array
	
	def hits(self):
		total = self.sum().values.item()
		return total
	

class MatchingOperator:
	pass

class BinaryProximityOperator(MatchingOperator):
	'''
	Produced a simple binary contingency table where the score has value 
	which is less than or equal to the specified tolerance, based on values
	rounded to a specific numerical precision. Due to floating-point
	errors, it is important to specify an appropriate precision.

	If your data is likely to have significance at extremely high precision, 
	floating point values are not appropriate to use, and something like the
	python decimal.Decimal should be utilized. 
	'''

	def __init__(self, *, precision=DEFAULT_PRECISION, tolerance=0):
		self.precision = precision
		self.tolerance = tolerance

	def make_table(self, proximity):
		proximity = proximity.round(self.precision)
		binaryArray = (proximity <= self.tolerance)
		result = BinaryTable(binaryArray)
		return result

class SimpleBucketOperator(MatchingOperator):
	def __init__(self, *, precision=DEFAULT_PRECISION, tolerance=0):
		pass