from textwrap import dedent
from peewee import ForeignKeyField, IntegerField, CharField, BooleanField
from playhouse.sqlite_ext import JSONField
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy

class Splitset(BaseModel):
	"""
	- Here the `samples_` attributes contain indices.
	-ToDo: store and visualize distributions of each column in training split, including label.
	-Future: is it useful to specify the size of only test for unsupervised learning?
	"""
	samples = JSONField()
	sizes = JSONField()
	supervision = CharField()
	has_test = BooleanField()
	has_validation = BooleanField()
	bin_count = IntegerField(null=True)
	unsupervised_stratify_col = CharField(null=True)

	label = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='splitsets')
	# Featureset is a many-to-many relationship between Splitset and Feature.

	def make(
		feature_ids:list
		, label_id:int = None
		, size_test:float = None
		, size_validation:float = None
		, bin_count:float = None
		, unsupervised_stratify_col:str = None
	):
		# The first feature_id is used for stratification, so it's best to use Tabular data in this slot.
		# --- Verify splits ---
		if (size_test is not None):
			if ((size_test <= 0.0) or (size_test >= 1.0)):
				raise ValueError("\nYikes - `size_test` must be between 0.0 and 1.0\n")
			# Don't handle `has_test` here. Need to check label first.
		
		if ((size_validation is not None) and (size_test is None)):
			raise ValueError("\nYikes - you specified a `size_validation` without setting a `size_test`.\n")

		if (size_validation is not None):
			if ((size_validation <= 0.0) or (size_validation >= 1.0)):
				raise ValueError("\nYikes - `size_test` must be between 0.0 and 1.0\n")
			sum_test_val = size_validation + size_test
			if sum_test_val >= 1.0:
				raise ValueError("\nYikes - Sum of `size_test` + `size_test` must be between 0.0 and 1.0 to leave room for training set.\n")
			"""
			Have to run train_test_split twice do the math to figure out the size of 2nd split.
			Let's say I want {train:0.67, validation:0.13, test:0.20}
			The first test_size is 20% which leaves 80% of the original data to be split into validation and training data.
			(1.0/(1.0-0.20))*0.13 = 0.1625
			"""
			pct_for_2nd_split = (1.0/(1.0-size_test))*size_validation
			has_validation = True
		else:
			has_validation = False

		# --- Verify features ---
		feature_ids = listify(feature_ids)
		feature_lengths = []
		for f_id in feature_ids:
			f = Feature.get_by_id(f_id)
			f_dataset = f.dataset
			f_dset_type = f_dataset.dataset_type

			if (f_dset_type == 'tabular' or f_dset_type == 'text'):
				f_length = Dataset.get_main_file(f_dataset.id).shape['rows']
			elif (f_dset_type == 'image' or f_dset_type == 'sequence'):
				f_length = f_dataset.file_count
			feature_lengths.append(f_length)
		if (len(set(feature_lengths)) != 1):
			raise ValueError("Yikes - List of features you provided contain different amounts of samples: {set(feature_lengths)}")

		# --- Prepare for splitting ---
		feature = Feature.get_by_id(feature_ids[0])
		f_dataset = feature.dataset
		f_dset_type = f_dataset.dataset_type
		f_cols = feature.columns
		"""
		Simulate an index to be split alongside features and labels
		in order to keep track of the samples being used in the resulting splits.
		"""
		if (f_dset_type=='tabular' or f_dset_type=='text' or f_dset_type=='sequence'):
			# Could get the row count via `f_dataset.get_main_file().shape['rows']`, but need array later.
			feature_array = f_dataset.to_numpy(columns=f_cols) #Used below for splitting.
			# Works on both 2D and 3D data.
			sample_count = feature_array.shape[0]
		elif (f_dset_type=='image'):
			sample_count = f_dataset.file_count
		arr_idx = np.arange(sample_count)
		
		samples = {}
		sizes = {}
		if (size_test is None):
			size_test = 0.30

		# ------ Stratification prep ------
		if (label_id is not None):
			has_test = True
			supervision = "supervised"
			if (unsupervised_stratify_col is not None):
				raise ValueError("\nYikes - `unsupervised_stratify_col` cannot be present is there is a Label.\n")

			# We don't need to prevent duplicate Label/Feature combos because Splits generate different samples each time.
			label = Label.get_by_id(label_id)
			# Check number of samples in Label vs Feature, because they can come from different Datasets.
			stratify_arr = label.to_numpy()
			l_length = label.dataset.get_main_file().shape['rows']
			
			if (label.dataset.id != f_dataset.id):
				if (l_length != sample_count):
					raise ValueError("\nYikes - The Datasets of your Label and Feature do not contains the same number of samples.\n")

			
			# check for OHE cols and reverse them so we can still stratify ordinally.
			if (stratify_arr.shape[1] > 1):
				stratify_arr = np.argmax(stratify_arr, axis=1)
			# OHE dtype returns as int64
			stratify_dtype = stratify_arr.dtype


		elif (label_id is None):
			has_test = False
			supervision = "unsupervised"
			label = None

			indices_lst_train = arr_idx.tolist()

			if (unsupervised_stratify_col is not None):
				if (f_dset_type=='image'):
					raise ValueError("\nYikes - `unsupervised_stratify_col` cannot be used with `dataset_type=='image'`.\n")

				column_names = f_dataset.get_main_tabular().columns
				col_index = colIndices_from_colNames(column_names=column_names, desired_cols=[unsupervised_stratify_col])[0]
				stratify_arr = feature_array[:,:,col_index]
				stratify_dtype = stratify_arr.dtype
				if (f_dset_type=='sequence'):	
					if (stratify_arr.shape[1] > 1):
						# We need a single value, so take the median or mode of each 1D array.
						if (np.issubdtype(stratify_dtype, np.number) == True):
							stratify_arr = np.median(stratify_arr, axis=1)
						if (np.issubdtype(stratify_dtype, np.number) == False):
							modes = [scipy.stats.mode(arr1D)[0][0] for arr1D in stratify_arr]
							stratify_arr = np.array(modes)
						# Now both are 1D so reshape to 2D.
						stratify_arr = stratify_arr.reshape(stratify_arr.shape[0], 1)

			elif (unsupervised_stratify_col is None):
				if (bin_count is not None):
		 			raise ValueError("\nYikes - `bin_count` cannot be set if `unsupervised_stratify_col is None` and `label_id is None`.\n")
				stratify_arr = None#Used in if statements below.


		# ------ Stratified vs Unstratified ------		
		if (stratify_arr is not None):
			"""
			- `sklearn.model_selection.train_test_split` = https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
			- `shuffle` happens before the split. Although preserves a df's original index, we don't need to worry about that because we are providing our own indices.
			- Don't include the Dataset.Image.feature pixel arrays in stratification.
			"""
			# `bin_count` is only returned so that we can persist it.
			stratifier1, bin_count = Splitset.stratifier_by_dtype_binCount(
				stratify_dtype = stratify_dtype,
				stratify_arr = stratify_arr,
				bin_count = bin_count
			)

			if (f_dset_type=='tabular' or f_dset_type=='text' or f_dset_type=='sequence'):
				features_train, features_test, stratify_train, stratify_test, indices_train, indices_test = train_test_split(
					feature_array, stratify_arr, arr_idx
					, test_size = size_test
					, stratify = stratifier1
					, shuffle = True
				)

				if (size_validation is not None):
					stratifier2, bin_count = Splitset.stratifier_by_dtype_binCount(
						stratify_dtype = stratify_dtype,
						stratify_arr = stratify_train, #This split is different from stratifier1.
						bin_count = bin_count
					)

					features_train, features_validation, stratify_train, stratify_validation, indices_train, indices_validation = train_test_split(
						features_train, stratify_train, indices_train
						, test_size = pct_for_2nd_split
						, stratify = stratifier2
						, shuffle = True
					)

			elif (f_dset_type=='image'):
				# Differs in that the Features not fed into `train_test_split()`.
				stratify_train, stratify_test, indices_train, indices_test = train_test_split(
					stratify_arr, arr_idx
					, test_size = size_test
					, stratify = stratifier1
					, shuffle = True
				)

				if (size_validation is not None):
					stratifier2, bin_count = Splitset.stratifier_by_dtype_binCount(
						stratify_dtype = stratify_dtype,
						stratify_arr = stratify_train, #This split is different from stratifier1.
						bin_count = bin_count
					)

					stratify_train, stratify_validation, indices_train, indices_validation = train_test_split(
						stratify_train, indices_train
						, test_size = pct_for_2nd_split
						, stratify = stratifier2
						, shuffle = True
					)

		elif (stratify_arr is None):
			if (f_dset_type=='tabular' or f_dset_type=='text' or f_dset_type=='sequence'):
				features_train, features_test, indices_train, indices_test = train_test_split(
					feature_array, arr_idx
					, test_size = size_test
					, shuffle = True
				)

				if (size_validation is not None):
					features_train, features_validation, indices_train, indices_validation = train_test_split(
						features_train, indices_train
						, test_size = pct_for_2nd_split
						, shuffle = True
					)

			elif (f_dset_type=='image'):
				# Differs in that the Features not fed into `train_test_split()`.
				indices_train, indices_test = train_test_split(
					arr_idx
					, test_size = size_test
					, shuffle = True
				)

				if (size_validation is not None):
					indices_train, indices_validation = train_test_split(
						indices_train
						, test_size = pct_for_2nd_split
						, shuffle = True
					)

		
		if (size_validation is not None):
			indices_lst_validation = indices_validation.tolist()
			samples["validation"] = indices_lst_validation	

		indices_lst_train, indices_lst_test  = indices_train.tolist(), indices_test.tolist()
		samples["train"] = indices_lst_train
		samples["test"] = indices_lst_test

		size_train = 1.0 - size_test
		if (size_validation is not None):
			size_train -= size_validation
			count_validation = len(indices_lst_validation)
			sizes["validation"] =  {"percent": size_validation, "count": count_validation}
		
		count_test = len(indices_lst_test)
		count_train = len(indices_lst_train)
		sizes["test"] = {"percent": size_test, "count": count_test}
		sizes["train"] = {"percent": size_train, "count": count_train}


		splitset = Splitset.create(
			label = label
			, samples = samples
			, sizes = sizes
			, supervision = supervision
			, has_test = has_test
			, has_validation = has_validation
			, bin_count = bin_count
			, unsupervised_stratify_col = unsupervised_stratify_col
		)

		try:
			for f_id in feature_ids:
				feature = Feature.get_by_id(f_id)
				Featureset.create(splitset=splitset, feature=feature)
		except:
			splitset.delete_instance() # Orphaned.
			raise
		return splitset


	def values_to_bins(array_to_bin:object, bin_count:int):
		"""
		Overwites continuous Label values with bin numbers for statification & folding.
		Switched to `pd.qcut` because `np.digitize` never had enough samples in the up the leftmost/right bin.
		"""
		# Make 1D for qcut.
		array_to_bin = array_to_bin.flatten()
		# For really unbalanced labels, I ran into errors where bin boundaries would be duplicates all the way down to 2 bins.
		# Setting `duplicates='drop'` to address this.
		bin_numbers = pd.qcut(x=array_to_bin, q=bin_count, labels=False, duplicates='drop')
		# Convert 1D array back to 2D for the rest of the program.
		bin_numbers = np.reshape(bin_numbers, (-1, 1))
		return bin_numbers


	def stratifier_by_dtype_binCount(stratify_dtype:object, stratify_arr:object, bin_count:int=None):
		# Based on the dtype and bin_count determine how to stratify.
		# Automatically bin floats.
		if np.issubdtype(stratify_dtype, np.floating):
			if (bin_count is None):
				bin_count = 3
			stratifier = Splitset.values_to_bins(array_to_bin=stratify_arr, bin_count=bin_count)
		# Allow ints to pass either binned or unbinned.
		elif (
			(np.issubdtype(stratify_dtype, np.signedinteger))
			or
			(np.issubdtype(stratify_dtype, np.unsignedinteger))
		):
			if (bin_count is not None):
				stratifier = Splitset.values_to_bins(array_to_bin=stratify_arr, bin_count=bin_count)
			elif (bin_count is None):
				# Assumes the int is for classification.
				stratifier = stratify_arr
		# Reject binned objs.
		elif (np.issubdtype(stratify_dtype, np.number) == False):
			if (bin_count is not None):
				raise ValueError(dedent("""
					Yikes - Your Label is not numeric (neither `np.floating`, `np.signedinteger`, `np.unsignedinteger`).
					Therefore, you cannot provide a value for `bin_count`.
				\n"""))
			elif (bin_count is None):
				stratifier = stratify_arr

		return stratifier, bin_count


	def get_features(id:int):
		splitset = Splitset.get_by_id(id)
		features = list(Feature.select().join(Featureset).where(Featureset.splitset==splitset))
		return features


	def make_foldset(
		id:int
		, fold_count:int = None
		, bin_count:int = None
	):
		foldset = Foldset.from_splitset(
			splitset_id = id
			, fold_count = fold_count
			, bin_count = bin_count
		)
		return foldset