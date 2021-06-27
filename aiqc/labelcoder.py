import inspect, warnings
from textwrap import dedent
from peewee import CharField, BooleanField, ForeignKeyField
from playhouse.fields import PickleField
from playhouse.sqlite_ext import JSONField
from .basemodel import BaseModel
from .label import Label
from .labelcoder import Labelcoder
import constants
import numpy as np

class Labelcoder(BaseModel):
	"""
	- `is_fit_train` toggles if the encoder is either `.fit(<training_split/fold>)` to 
	  avoid bias or `.fit(<entire_dataset>)`.
	- Categorical (ordinal and OHE) encoders are best applied to entire dataset in case 
	  there are classes missing in the split/folds of validation/ test data.
	- Whereas numerical encoders are best fit only to the training data.
	- Because there's only 1 encoder that runs and it uses all columns, Labelcoder 
	  is much simpler to validate and run in comparison to Featurecoder.
	"""
	only_fit_train = BooleanField()
	is_categorical = BooleanField()
	sklearn_preprocess = PickleField()
	matching_columns = JSONField() # kinda unecessary, but maybe multi-label future.
	encoding_dimension = CharField()

	label = ForeignKeyField(Label, backref='labelcoders')

	def from_label(
		label_id:int
		, sklearn_preprocess:object
	):
		label = Label.get_by_id(label_id)

		sklearn_preprocess, only_fit_train, is_categorical = Labelcoder.check_sklearn_attributes(
			sklearn_preprocess, is_label=True
		)

		samples_to_encode = label.to_numpy()
		# 2. Test Fit.
		try:
			fitted_encoders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
				sklearn_preprocess = sklearn_preprocess
				, samples_to_fit = samples_to_encode
			)
		except:
			print(f"\nYikes - During a test encoding, failed to `fit()` instantiated `{sklearn_preprocess}` on `label.to_numpy())`.\n")
			raise

		# 3. Test Transform/ Encode.
		try:
			"""
			- During `Job.run`, it will touch every split/fold regardless of what it was fit on
			  so just validate it on whole dataset.
			"""
			Labelcoder.transform_dynamicDimensions(
				fitted_encoders = fitted_encoders
				, encoding_dimension = encoding_dimension
				, samples_to_transform = samples_to_encode
			)
		except:
			raise ValueError(dedent("""
			During testing, the encoder was successfully `fit()` on the labels,
			but, it failed to `transform()` labels of the dataset as a whole.
			"""))
		else:
			pass    
		lc = Labelcoder.create(
			only_fit_train = only_fit_train
			, sklearn_preprocess = sklearn_preprocess
			, encoding_dimension = encoding_dimension
			, matching_columns = label.columns
			, is_categorical = is_categorical
			, label = label
		)
		return lc


	def check_sklearn_attributes(sklearn_preprocess:object, is_label:bool):
		#This function is used by Featurecoder too, so don't put label-specific things in here.

		if (inspect.isclass(sklearn_preprocess)):
			raise ValueError(dedent("""
				Yikes - The encoder you provided is a class name, but it should be a class instance.\n
				Class (incorrect): `OrdinalEncoder`
				Instance (correct): `OrdinalEncoder()`
			"""))

		# Encoder parent modules vary: `sklearn.preprocessing._data` vs `sklearn.preprocessing._label`
		# Feels cleaner than this: https://stackoverflow.com/questions/14570802/python-check-if-object-is-instance-of-any-class-from-a-certain-module
		coder_type = str(type(sklearn_preprocess))
		if ('sklearn.preprocessing' not in coder_type):
			raise ValueError(dedent("""
				Yikes - At this point in time, only `sklearn.preprocessing` encoders are supported.
				https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
			"""))
		elif ('sklearn.preprocessing' in coder_type):
			if (not hasattr(sklearn_preprocess, 'fit')):    
				raise ValueError(dedent("""
					Yikes - The `sklearn.preprocessing` method you provided does not have a `fit` method.\n
					Please use one of the uppercase methods instead.
					For example: use `RobustScaler` instead of `robust_scale`.
				"""))

			if (hasattr(sklearn_preprocess, 'sparse')):
				if (sklearn_preprocess.sparse == True):
					try:
						sklearn_preprocess.sparse = False
						print(dedent("""
							=> Info - System overriding user input to set `sklearn_preprocess.sparse=False`.
							   This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.
						"""))
					except:
						raise ValueError(dedent(f"""
							Yikes - Detected `sparse==True` attribute of {sklearn_preprocess}.
							System attempted to override this to False, but failed.
							FYI `sparse` is True by default if left blank.
							This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.\n
							Please try again with False. For example, `OneHotEncoder(sparse=False)`.
						"""))

			if (hasattr(sklearn_preprocess, 'copy')):
				if (sklearn_preprocess.copy == True):
					try:
						sklearn_preprocess.copy = False
						print(dedent("""
							=> Info - System overriding user input to set `sklearn_preprocess.copy=False`.
							   This saves memory when concatenating the output of many encoders.
						"""))
					except:
						raise ValueError(dedent(f"""
							Yikes - Detected `copy==True` attribute of {sklearn_preprocess}.
							System attempted to override this to False, but failed.
							FYI `copy` is True by default if left blank, which consumes memory.\n
							Please try again with 'copy=False'.
							For example, `StandardScaler(copy=False)`.
						"""))
			
			if (hasattr(sklearn_preprocess, 'sparse_output')):
				if (sklearn_preprocess.sparse_output == True):
					try:
						sklearn_preprocess.sparse_output = False
						print(dedent("""
							=> Info - System overriding user input to set `sklearn_preprocess.sparse_output=False`.
							   This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.
						"""))
					except:
						raise ValueError(dedent(f"""
							Yikes - Detected `sparse_output==True` attribute of {sklearn_preprocess}.
							System attempted to override this to True, but failed.
							Please try again with 'sparse_output=False'.
							This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.\n
							For example, `LabelBinarizer(sparse_output=False)`.
						"""))

			if (hasattr(sklearn_preprocess, 'order')):
				if (sklearn_preprocess.order == 'F'):
					try:
						sklearn_preprocess.order = 'C'
						print(dedent("""
							=> Info - System overriding user input to set `sklearn_preprocess.order='C'`.
							   This changes the output shape of the 
						"""))
					except:
						raise ValueError(dedent(f"""
							System attempted to override this to 'C', but failed.
							Yikes - Detected `order=='F'` attribute of {sklearn_preprocess}.
							Please try again with 'order='C'.
							For example, `PolynomialFeatures(order='C')`.
						"""))

			if (hasattr(sklearn_preprocess, 'encode')):
				if (sklearn_preprocess.encode == 'onehot'):
					# Multiple options here, so don't override user input.
					raise ValueError(dedent(f"""
						Yikes - Detected `encode=='onehot'` attribute of {sklearn_preprocess}.
						FYI `encode` is 'onehot' by default if left blank and it predictors in 'scipy.sparse.csr.csr_matrix',
						which causes Keras training to fail.\n
						Please try again with 'onehot-dense' or 'ordinal'.
						For example, `KBinsDiscretizer(encode='onehot-dense')`.
					"""))

			if (
				(is_label==True)
				and
				(not hasattr(sklearn_preprocess, 'inverse_transform'))
			):
				print(dedent("""
					Warning - The following encoders do not have an `inverse_transform` method.
					It is inadvisable to use them to encode Labels during training, 
					because you may not be able to programmatically decode your raw predictions 
					when it comes time for inference (aka non-training predictions):

					[Binarizer, KernelCenterer, Normalizer, PolynomialFeatures]
				"""))

			"""
			- Binners like 'KBinsDiscretizer' and 'QuantileTransformer'
			  will place unseen observations outside bounds into existing min/max bin.
			- I assume that someone won't use a custom FunctionTransformer, for categories
			  when all of these categories are available.
			- LabelBinarizer is not threshold-based, it's more like an OHE.
			"""
			only_fit_train = True
			stringified_coder = str(sklearn_preprocess)
			is_categorical = False
			for c in constants.categorical_encoders:
				if (stringified_coder.startswith(c)):
					only_fit_train = False
					is_categorical = True
					break

			return sklearn_preprocess, only_fit_train, is_categorical


	def fit_dynamicDimensions(sklearn_preprocess:object, samples_to_fit:object):
		"""
		- Future: optimize to make sure not duplicating numpy. especially append to lists + reshape after transpose.
		- There are 17 uppercase sklearn encoders, and 10 different data types across float, str, int 
		  when consider negatives, 2D multiple columns, 2D single columns.
		- Different encoders work with different data types and dimensionality.
		- This function normalizes that process by coercing the dimensionality that the encoder wants,
		  and erroring if the wrong data type is used. The goal in doing so is to return 
		  that dimensionality for future use.

		- `samples_to_transform` is pre-filtered for the appropriate `matching_columns`.
		- The rub lies in that if you have many columns, but the encoder only fits 1 column at a time, 
		  then you return many fits for a single type of preprocess.
		- Remember this is for a single Featurecoder that is potential returning multiple fits.

		- UPDATE: after disabling LabelBinarizer and LabelEncoder from running on multiple columns,
		  everything seems to be fitting as "2D_multiColumn", but let's keep the logic for new sklearn methods.
		"""
		fitted_encoders = []
		incompatibilities = {
			"string": [
				"KBinsDiscretizer", "KernelCenterer", "MaxAbsScaler", 
				"MinMaxScaler", "PowerTransformer", "QuantileTransformer", 
				"RobustScaler", "StandardScaler"
			]
			, "float": ["LabelBinarizer"]
			, "numeric array without dimensions both odd and square (e.g. 3x3, 5x5)": ["KernelCenterer"]
		}

		with warnings.catch_warnings(record=True) as w:
			try:
				# aiqc `to_numpy()` always fetches 2D.
				# Remember, we are assembling `fitted_encoders` dict, not accesing it.
				fit_encoder = sklearn_preprocess.fit(samples_to_fit)
				fitted_encoders.append(fit_encoder)
			except:
				# At this point, "2D" failed. It had 1 or more columns.
				try:
					width = samples_to_fit.shape[1]
					if (width > 1):
						# Reshape "2D many columns" to “3D of 2D single columns.”
						samples_to_fit = samples_to_fit[None].T                    
						# "2D single column" already failed. Need it to fail again to trigger except.
					elif (width == 1):
						# Reshape "2D single columns" to “3D of 2D single columns.”
						samples_to_fit = samples_to_fit.reshape(1, samples_to_fit.shape[0], 1)    
					# Fit against each 2D array within the 3D array.
					for i, arr in enumerate(samples_to_fit):
						fit_encoder = sklearn_preprocess.fit(arr)
						fitted_encoders.append(fit_encoder)
				except:
					# At this point, "2D single column" has failed.
					try:
						# So reshape the "3D of 2D_singleColumn" into "2D of 1D for each column."
						# This transformation is tested for both (width==1) as well as (width>1). 
						samples_to_fit = samples_to_fit.transpose(2,0,1)[0]
						# Fit against each column in 2D array.
						for i, arr in enumerate(samples_to_fit):
							fit_encoder = sklearn_preprocess.fit(arr)
							fitted_encoders.append(fit_encoder)
					except:
						raise ValueError(dedent(f"""
							Yikes - Encoder failed to fit the columns you filtered.\n
							Either the data is dirty (e.g. contains NaNs),
							or the encoder might not accept negative values (e.g. PowerTransformer.method='box-cox'),
							or you used one of the incompatible combinations of data type and encoder seen below:\n
							{incompatibilities}
						"""))
					else:
						encoding_dimension = "1D"
				else:
					encoding_dimension = "2D_singleColumn"
			else:
				encoding_dimension = "2D_multiColumn"
		return fitted_encoders, encoding_dimension


	def if_1d_make_2d(array:object):
		if (len(array.shape) == 1):
			array = array.reshape(array.shape[0], 1)
		return array


	def transform_dynamicDimensions(
		fitted_encoders:list
		, encoding_dimension:str
		, samples_to_transform:object
	):
		"""
		- UPDATE: after disabling LabelBinarizer and LabelEncoder from running on multiple columns,
		  everything seems to be fitting as "2D_multiColumn", but let's keep the logic for new sklearn methods.
		"""
		if (encoding_dimension == '2D_multiColumn'):
			# Our `to_numpy` method fetches data as 2D. So it has 1+ columns. 
			encoded_samples = fitted_encoders[0].transform(samples_to_transform)
			encoded_samples = Labelcoder.if_1d_make_2d(array=encoded_samples)
		elif (encoding_dimension == '2D_singleColumn'):
			# Means that `2D_multiColumn` arrays cannot be used as is.
			width = samples_to_transform.shape[1]
			if (width == 1):
				# It's already "2D_singleColumn"
				encoded_samples = fitted_encoders[0].transform(samples_to_transform)
				encoded_samples = Labelcoder.if_1d_make_2d(array=encoded_samples)
			elif (width > 1):
				# Data must be fed into encoder as separate '2D_singleColumn' arrays.
				# Reshape "2D many columns" to “3D of 2D singleColumns” so we can loop on it.
				encoded_samples = samples_to_transform[None].T
				encoded_arrs = []
				for i, arr in enumerate(encoded_samples):
					encoded_arr = fitted_encoders[i].transform(arr)
					encoded_arr = Labelcoder.if_1d_make_2d(array=encoded_arr)  
					encoded_arrs.append(encoded_arr)
				encoded_samples = np.array(encoded_arrs).T

				# From "3D of 2Ds" to "2D wide"
				# When `encoded_samples` was accidentally a 3D shape, this fixed it:
				"""
				if (len(encoded_samples.shape) == 3):
					encoded_samples = encoded_samples.transpose(
						1,0,2
					).reshape(
						# where index represents dimension.
						encoded_samples.shape[1],
						encoded_samples.shape[0]*encoded_samples.shape[2]
					)
				"""
				del encoded_arrs
		elif (encoding_dimension == '1D'):
			# From "2D_multiColumn" to "2D with 1D for each column"
			# This `.T` works for both single and multi column.
			encoded_samples = samples_to_transform.T
			# Since each column is 1D, we care about rows now.
			length = encoded_samples.shape[0]
			if (length == 1):
				encoded_samples = fitted_encoders[0].transform(encoded_samples)
				# Some of these 1D encoders also output 1D.
				# Need to put it back into 2D.
				encoded_samples = Labelcoder.if_1d_make_2d(array=encoded_samples)  
			elif (length > 1):
				encoded_arrs = []
				for i, arr in enumerate(encoded_samples):
					encoded_arr = fitted_encoders[i].transform(arr)
					# Check if it is 1D before appending.
					encoded_arr = Labelcoder.if_1d_make_2d(array=encoded_arr)              
					encoded_arrs.append(encoded_arr)
				# From "3D of 2D_singleColumn" to "2D_multiColumn"
				encoded_samples = np.array(encoded_arrs).T
				del encoded_arrs
		return encoded_samples