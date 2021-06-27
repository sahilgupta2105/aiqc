import pprint
from textwrap import dedent
from peewee import CharField, IntegerField, BooleanField, ForeignKeyField, JSONField, PickleField
from .basemodel import BaseModel
from .encoderset import Encoderset
from .utility import listify
from .labelcoder import Labelcoder

class Featurecoder(BaseModel):
	"""
	- An Encoderset can have a chain of Featurecoders.
	- Encoders are applied sequential, meaning the columns encoded by `featurecoder_index=0` 
	  are not available to `featurecoder_index=1`.
	- Much validation because real-life encoding errors are cryptic and deep for beginners.
	"""
	featurecoder_index = IntegerField()
	sklearn_preprocess = PickleField()
	matching_columns = JSONField()
	leftover_columns = JSONField()
	leftover_dtypes = JSONField()
	original_filter = JSONField()
	encoding_dimension = CharField()
	only_fit_train = BooleanField()
	is_categorical = BooleanField()

	encoderset = ForeignKeyField(Encoderset, backref='featurecoders')


	def from_encoderset(
		encoderset_id:int
		, sklearn_preprocess:object
		, include:bool = True
		, dtypes:list = None
		, columns:list = None
		, verbose:bool = True
	):
		encoderset = Encoderset.get_by_id(encoderset_id)
		dtypes = listify(dtypes)
		columns = listify(columns)
		
		feature = encoderset.feature
		feature_cols = feature.columns
		feature_dtypes = feature.get_dtypes()
		existing_featurecoders = list(encoderset.featurecoders)

		dataset = feature.dataset
		dataset_type = dataset.dataset_type

		# 1. Figure out which columns have yet to be encoded.
		# Order-wise no need to validate filters if there are no columns left to filter.
		# Remember Feature columns are a subset of the Dataset columns.
		if (len(existing_featurecoders) == 0):
			initial_columns = feature_cols
			featurecoder_index = 0
		elif (len(existing_featurecoders) > 0):
			# Get the leftover columns from the last one.
			initial_columns = existing_featurecoders[-1].leftover_columns

			featurecoder_index = existing_featurecoders[-1].featurecoder_index + 1
			if (len(initial_columns) == 0):
				raise ValueError("\nYikes - All features already have encoders associated with them. Cannot add more Featurecoders to this Encoderset.\n")
		initial_dtypes = {}
		for key,value in feature_dtypes.items():
			for col in initial_columns:
				if (col == key):
					initial_dtypes[col] = value
					# Exit `c` loop early becuase matching `c` found.
					break

		if (verbose == True):
			print(f"\n___/ featurecoder_index: {featurecoder_index} \\_________\n") # Intentionally no trailing `\n`.

		# 2. Validate the lists of dtypes and columns provided as filters.
		if (dataset_type == "image"):
			raise ValueError("\nYikes - `Dataset.dataset_type=='image'` does not support encoding Feature.\n")
		
		sklearn_preprocess, only_fit_train, is_categorical = Labelcoder.check_sklearn_attributes(
			sklearn_preprocess, is_label=False
		)

		if (dtypes is not None):
			for typ in dtypes:
				if (typ not in set(initial_dtypes.values())):
					raise ValueError(dedent(f"""
					Yikes - dtype '{typ}' was not found in remaining dtypes.
					Remove '{typ}' from `dtypes` and try again.
					"""))
		
		if (columns is not None):
			for c in columns:
				if (col not in initial_columns):
					raise ValueError(dedent(f"""
					Yikes - Column '{col}' was not found in remaining columns.
					Remove '{col}' from `columns` and try again.
					"""))
		
		# 3a. Figure out which columns the filters apply to.
		if (include==True):
			# Add to this empty list via inclusion.
			matching_columns = []
			
			if ((dtypes is None) and (columns is None)):
				raise ValueError("\nYikes - When `include==True`, either `dtypes` or `columns` must be provided.\n")

			if (dtypes is not None):
				for typ in dtypes:
					for key,value in initial_dtypes.items():
						if (value == typ):
							matching_columns.append(key)
							# Don't `break`; there can be more than one match.

			if (columns is not None):
				for c in columns:
					# Remember that the dtype has already added some columns.
					if (c not in matching_columns):
						matching_columns.append(c)
					elif (c in matching_columns):
						# We know from validation above that the column existed in initial_columns.
						# Therefore, if it no longer exists it means that dtype_exclude got to it first.
						raise ValueError(dedent(f"""
						Yikes - The column '{c}' was already included by `dtypes`, so this column-based filter is not valid.
						Remove '{c}' from `columns` and try again.
						"""))

		elif (include==False):
			# Prune this list via exclusion.
			matching_columns = initial_columns.copy()

			if (dtypes is not None):
				for typ in dtypes:
					for key,value in initial_dtypes.items():                
						if (value == typ):
							matching_columns.remove(key)
							# Don't `break`; there can be more than one match.
			if (columns is not None):
				for c in columns:
					# Remember that the dtype has already pruned some columns.
					if (c in matching_columns):
						matching_columns.remove(c)
					elif (c not in matching_columns):
						# We know from validation above that the column existed in initial_columns.
						# Therefore, if it no longer exists it means that dtype_exclude got to it first.
						raise ValueError(dedent(f"""
						Yikes - The column '{c}' was already excluded by `dtypes`,
						so this column-based filter is not valid.
						Remove '{c}' from `dtypes` and try again.
						"""))
		if (len(matching_columns) == 0):
			if (include == True):
				inex_str = "inclusion"
			elif (include == False):
				inex_str = "exclusion"
			raise ValueError(f"\nYikes - There are no columns left to use after applying the dtype and column {inex_str} filters.\n")
		elif (
			(
				(str(sklearn_preprocess).startswith("LabelBinarizer"))
				or 
				(str(sklearn_preprocess).startswith("LabelEncoder"))
			)
			and
			(len(matching_columns) > 1)
		):
			raise ValueError(dedent("""
				Yikes - `LabelBinarizer` or `LabelEncoder` cannot be run on 
				multiple columns at once.

				We have frequently observed inconsistent behavior where they 
				often ouput incompatible array shapes that cannot be scalable 
				concatenated, or they succeed in fitting, but fail at transforming.
				
				We recommend you either use these with 1 column at a 
				time or switch to another encoder.
			"""))

		# 3b. Record the  output.
		leftover_columns =  list(set(initial_columns) - set(matching_columns))
		# This becomes leftover_dtypes.
		for c in matching_columns:
			del initial_dtypes[c]

		original_filter = {
			'include': include
			, 'dtypes': dtypes
			, 'columns': columns
		}

		# 4. Test fitting the encoder to matching columns.
		samples_to_encode = feature.to_numpy(columns=matching_columns)
		# Handles `Dataset.Sequence` by stacking the 2D arrays into a tall 2D array.
		features_shape = samples_to_encode.shape
		if (len(features_shape)==3):
			rows_2D = features_shape[0] * features_shape[1]
			samples_to_encode = samples_to_encode.reshape(rows_2D, features_shape[2])

		fitted_encoders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
			sklearn_preprocess = sklearn_preprocess
			, samples_to_fit = samples_to_encode
		)

		# 5. Test encoding the whole dataset using fitted encoder on matching columns.
		try:
			Labelcoder.transform_dynamicDimensions(
				fitted_encoders = fitted_encoders
				, encoding_dimension = encoding_dimension
				, samples_to_transform = samples_to_encode
			)
		except:
			raise ValueError(dedent("""
			During testing, the encoder was successfully `fit()` on the features,
			but, it failed to `transform()` features of the dataset as a whole.\n
			"""))
		else:
			pass

		featurecoder = Featurecoder.create(
			featurecoder_index = featurecoder_index
			, only_fit_train = only_fit_train
			, is_categorical = is_categorical
			, sklearn_preprocess = sklearn_preprocess
			, matching_columns = matching_columns
			, leftover_columns = leftover_columns
			, leftover_dtypes = initial_dtypes#pruned
			, original_filter = original_filter
			, encoderset = encoderset
			, encoding_dimension = encoding_dimension
		)

		if (verbose == True):
			print(
				f"=> The column(s) below matched your filter(s) and were ran through a test-encoding successfully.\n\n" \
				f"{matching_columns}\n" 
			)
			if (len(leftover_columns) == 0):
				print(
					f"=> Done. All feature column(s) have encoder(s) associated with them.\n" \
					f"No more Featurecoders can be added to this Encoderset.\n"
				)
			elif (len(leftover_columns) > 0):
				print(
					f"=> The remaining column(s) and dtype(s) can be used in downstream Featurecoder(s):\n" \
					f"{pprint.pformat(initial_dtypes)}\n"
				)
		return featurecoder