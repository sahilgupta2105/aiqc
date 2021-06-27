import math
from textwrap import dedent
from peewee import IntegerField, ForeignKeyField
from playhouse.sqlite_ext import JSONField
from .basemodel import BaseModel
from .dataset import Dataset
from .utility import listify
from .splitset import Splitset
from .encoderset import Encoderset

class Feature(BaseModel):
	"""
	- Remember, a Feature is just a record of the columns being used.
	- Decided not to go w subclasses of Unsupervised and Supervised because that would complicate the SDK for the user,
	  and it essentially forked every downstream model into two subclasses.
	- PCA components vary across features. When different columns are used those columns have different component values.
	"""
	columns = JSONField(null=True)
	columns_excluded = JSONField(null=True)
	dataset = ForeignKeyField(Dataset, backref='features')


	def from_dataset(
		dataset_id:int
		, include_columns:list=None
		, exclude_columns:list=None
		#Future: runPCA #,run_pca:boolean=False # triggers PCA analysis of all columns
	):
		"""
		As we get further away from the `Dataset.<Types>` they need less isolation.
		"""
		dataset = Dataset.get_by_id(dataset_id)
		include_columns = listify(include_columns)
		exclude_columns = listify(exclude_columns)

		if (dataset.dataset_type == 'image'):
			# Just passes the Dataset through for now.
			if (include_columns is not None) or (exclude_columns is not None):
				raise ValueError("\nYikes - The `Dataset.Image` classes supports neither the `include_columns` nor `exclude_columns` arguemnt.\n")
			columns = None
			columns_excluded = None
		elif (dataset.dataset_type == 'tabular' or dataset.dataset_type == 'text' or dataset.dataset_type == 'sequence'):
			d_cols = Dataset.get_main_tabular(dataset_id).columns

			if ((include_columns is not None) and (exclude_columns is not None)):
				raise ValueError("\nYikes - You can set either `include_columns` or `exclude_columns`, but not both.\n")

			if (include_columns is not None):
				# check columns exist
				all_cols_found = all(col in d_cols for col in include_columns)
				if (not all_cols_found):
					raise ValueError("\nYikes - You specified `include_columns` that do not exist in the Dataset.\n")
				# inclusion
				columns = include_columns
				# exclusion
				columns_excluded = d_cols
				for col in include_columns:
					columns_excluded.remove(col)

			elif (exclude_columns is not None):
				all_cols_found = all(col in d_cols for col in exclude_columns)
				if (not all_cols_found):
					raise ValueError("\nYikes - You specified `exclude_columns` that do not exist in the Dataset.\n")
				# exclusion
				columns_excluded = exclude_columns
				# inclusion
				columns = d_cols
				for col in exclude_columns:
					columns.remove(col)
				if (not columns):
					raise ValueError("\nYikes - You cannot exclude every column in the Dataset. For there will be nothing to analyze.\n")
			else:
				columns = d_cols
				columns_excluded = None

			"""
			- Check that this Dataset does not already have a Feature that is exactly the same.
			- There are less entries in `excluded_columns` so maybe it's faster to compare that.
			"""
			if columns_excluded is not None:
				cols_aplha = sorted(columns_excluded)
			else:
				cols_aplha = None
			d_features = dataset.features
			count = d_features.count()
			if (count > 0):
				for f in d_features:
					f_id = str(f.id)
					f_cols = f.columns_excluded
					if (f_cols is not None):
						f_cols_alpha = sorted(f_cols)
					else:
						f_cols_alpha = None
					if (cols_aplha == f_cols_alpha):
						raise ValueError(dedent(f"""
						Yikes - This Dataset already has Feature <id:{f_id}> with the same columns.
						Cannot create duplicate.
						"""))

		feature = Feature.create(
			dataset = dataset
			, columns = columns
			, columns_excluded = columns_excluded
		)
		return feature


	def to_pandas(id:int, samples:list=None, columns:list=None):
		samples = listify(samples)
		columns = listify(columns)
		f_frame = Feature.get_feature(
			id = id
			, numpy_or_pandas = 'pandas'
			, samples = samples
			, columns = columns
		)
		return f_frame


	def to_numpy(id:int, samples:list=None, columns:list=None):
		samples = listify(samples)
		columns = listify(columns)
		f_arr = Feature.get_feature(
			id = id
			, numpy_or_pandas = 'numpy'
			, samples = samples
			, columns = columns
		)
		return f_arr


	def get_feature(
		id:int
		, numpy_or_pandas:str
		, samples:list = None
		, columns:list = None
	):
		feature = Feature.get_by_id(id)
		samples = listify(samples)
		columns = listify(columns)
		f_cols = feature.columns

		if (columns is not None):
			for c in columns:
				if c not in f_cols:
					raise ValueError("\nYikes - Cannot fetch column '{c}' because it is not in `Feature.columns`.\n")
			f_cols = columns    

		dataset_id = feature.dataset.id

		if (numpy_or_pandas == 'numpy'):
			f_data = Dataset.to_numpy(
				id = dataset_id
				, columns = f_cols
				, samples = samples
			)
		elif (numpy_or_pandas == 'pandas'):
			f_data = Dataset.to_pandas(
				id = dataset_id
				, columns = f_cols
				, samples = samples
			)
		return f_data


	def get_dtypes(
		id:int
	):
		feature = Feature.get_by_id(id)
		dataset = feature.dataset
		if (dataset.dataset_type == 'image'):
			raise ValueError("\nYikes - `feature.dataset.dataset_type=='image'` does not have dtypes.\n")

		f_cols = feature.columns
		tabular_dtype = Dataset.get_main_tabular(dataset.id).dtypes

		feature_dtypes = {}
		for key,value in tabular_dtype.items():
			for col in f_cols:         
				if (col == key):
					feature_dtypes[col] = value
					# Exit `col` loop early becuase matching `col` found.
					break
		return feature_dtypes


	def make_splitset(
		id:int
		, label_id:int = None
		, size_test:float = None
		, size_validation:float = None
		, bin_count:int = None
		, unsupervised_stratify_col:str = None
	):
		splitset = Splitset.from_feature(
			feature_id = id
			, label_id = label_id
			, size_test = size_test
			, size_validation = size_validation
			, bin_count = bin_count
			, unsupervised_stratify_col = unsupervised_stratify_col
		)
		return splitset


	def make_encoderset(
		id:int
		, encoder_count:int = 0
		, description:str = None
	):
		encoderset = Encoderset.from_feature(
			feature_id = id
			, encoder_count = 0
			, description = description
		)
		return encoderset


	def get_latest_encoderset(id:int):
		feature = Feature.get_by_id(id)
		encodersets = list(feature.encodersets)
		# Check if list empty.
		if (not encodersets):
			return None
		else:
			return encodersets[-1]


	def make_window(id:int, size_window:int, size_shift:int):
		feature = Feature.get_by_id(id)
		window = Window.from_feature(
			size_window = size_window
			, size_shift = size_shift
			, feature_id = feature.id
		)
		return window




class Window(BaseModel):
	size_window = IntegerField()
	size_shift = IntegerField()
	feature = ForeignKeyField(Feature, backref='windows')


	def from_feature(
		feature_id:int
		, size_window:int
		, size_shift:int
	):
		feature = Feature.get_by_id(feature_id)
		file_count = feature.dataset.file_count

		if ((size_window < 1) or (size_window > (file_count - size_shift))):
			raise ValueError("\nYikes - Failed: `(size_window < 1) or (size_window > (file_count - size_shift)`.\n")
		if ((size_shift < 1) or (size_shift > (file_count - size_window))):
			raise ValueError("\nYikes - Failed: `(size_shift < 1) or (size_shift > (file_count - size_window)`.\n")

		window = Window.create(
			size_window = size_window
			, size_shift = size_shift
			, feature_id = feature.id
		)
		return window


	def shift_window_arrs(id:int, ndarray:object):
		window = Window.get_by_id(id)
		file_count = window.feature.dataset.file_count
		size_window = window.size_window
		size_shift = window.size_shift

		total_intervals = math.floor((file_count - size_shift) / size_window)

		#prune_shifted_lag = 0
		prune_shifted_lead = file_count - (total_intervals * size_window)
		prune_unshifted_lag = -(size_shift)
		prune_unshifted_lead = file_count - (total_intervals * size_window) - size_shift

		arr_shifted = arr_shifted = ndarray[prune_shifted_lead:]#:prune_shifted_lag
		arr_unshifted = ndarray[prune_unshifted_lead:prune_unshifted_lag]

		arr_shifted_shapes = arr_shifted.shape
		arr_shifted = arr_shifted.reshape(
			total_intervals#3D
			, arr_shifted_shapes[1]*math.floor(arr_shifted_shapes[0] / total_intervals)#rows
			, arr_shifted_shapes[2]#cols
		)
		arr_unshifted = arr_unshifted.reshape(
			total_intervals#3D
			, arr_shifted_shapes[1]*math.floor(arr_shifted_shapes[0] / total_intervals)#rows
			, arr_shifted_shapes[2]#cols
		)
		return arr_shifted, arr_unshifted


class Featureset(BaseModel):
	"""Featureset is a many-to-many relationship between Splitset and Feature."""
	splitset = ForeignKeyField(Splitset, backref='featuresets')
	feature = ForeignKeyField(Feature, backref='featuresets')