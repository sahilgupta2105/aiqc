import random
from textwrap import dedent
from peewee import ForeignKeyField, IntegerField, JSONField
from .basemodel import BaseModel
from .splitset import Splitset
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold, KFold

class Foldset(BaseModel):
	"""
	- Contains aggregate summary statistics and evaluate metrics for all Folds.
	- Works the same for all dataset types because only the labels are used for stratification.
	"""
	fold_count = IntegerField()
	random_state = IntegerField()
	bin_count = IntegerField(null=True) # For stratifying continuous features.
	#ToDo: max_samples_per_bin = IntegerField()
	#ToDo: min_samples_per_bin = IntegerField()

	splitset = ForeignKeyField(Splitset, backref='foldsets')

	def from_splitset(
		splitset_id:int
		, fold_count:int = None
		, bin_count:int = None
	):
		splitset = Splitset.get_by_id(splitset_id)
		new_random = False
		while new_random == False:
			random_state = random.randint(0, 4294967295) #2**32 - 1 inclusive
			matching_randoms = splitset.foldsets.select().where(Foldset.random_state==random_state)
			count_matches = matching_randoms.count()
			if count_matches == 0:
				new_random = True
		if (fold_count is None):
			fold_count = 5 # More likely than 4 to be evenly divisible.
		else:
			if (fold_count < 2):
				raise ValueError(dedent(f"""
				Yikes - Cross validation requires multiple folds.
				But you provided `fold_count`: <{fold_count}>.
				"""))
			elif (fold_count == 2):
				print("\nWarning - Instead of two folds, why not just use a validation split?\n")

		# Get the training indices. The actual values of the features don't matter, only label values needed for stratification.
		arr_train_indices = splitset.samples["train"]
		if (splitset.supervision=="supervised"):
			stratify_arr = splitset.label.to_numpy(samples=arr_train_indices)
			stratify_dtype = stratify_arr.dtype
		elif (splitset.supervision=="unsupervised"):
			if (splitset.unsupervised_stratify_col is not None):
				stratify_arr = splitset.get_features()[0].to_numpy(
					columns = splitset.unsupervised_stratify_col,
					samples = arr_train_indices
				)
				stratify_dtype = stratify_arr.dtype
				if (stratify_arr.shape[1] > 1):
					# We need a single value, so take the median or mode of each 1D array.
					if (np.issubdtype(stratify_dtype, np.number) == True):
						stratify_arr = np.median(stratify_arr, axis=1)
					if (np.issubdtype(stratify_dtype, np.number) == False):
						modes = [scipy.stats.mode(arr1D)[0][0] for arr1D in stratify_arr]
						stratify_arr = np.array(modes)
					# Now both are 1D so reshape to 2D.
					stratify_arr = stratify_arr.reshape(stratify_arr.shape[0], 1)
			elif (splitset.unsupervised_stratify_col is None):
				if (bin_count is not None):
					raise ValueError("\nYikes - `bin_count` cannot be set if `unsupervised_stratify_col is None` and `label_id is None`.\n")
				stratify_arr = None#Used in if statements below.
			
		# If the Labels are binned *overwite* the values w bin numbers. Otherwise untouched.
		if (stratify_arr is not None):
			# Bin the floats.
			if (np.issubdtype(stratify_dtype, np.floating)):
				if (bin_count is None):
					bin_count = splitset.bin_count #Inherit. 
				stratify_arr = Splitset.values_to_bins(
					array_to_bin = stratify_arr
					, bin_count = bin_count
				)
			# Allow ints to pass either binned or unbinned.
			elif (
				(np.issubdtype(stratify_dtype, np.signedinteger))
				or
				(np.issubdtype(stratify_dtype, np.unsignedinteger))
			):
				if (bin_count is not None):
					if (splitset.bin_count is None):
						print(dedent("""
							Warning - Previously you set `Splitset.bin_count is None`
							but now you are trying to set `Foldset.bin_count is not None`.
							
							This can result in incosistent stratification processes being 
							used for training samples versus validation and test samples.
						\n"""))
					stratify_arr = Splitset.values_to_bins(
						array_to_bin = stratify_arr
						, bin_count = bin_count
					)
			else:
				if (bin_count is not None):
					raise ValueError(dedent("""
						Yikes - The column you are stratifying by is not a numeric dtype (neither `np.floating`, `np.signedinteger`, `np.unsignedinteger`).
						Therefore, you cannot provide a value for `bin_count`.
					\n"""))

		train_count = len(arr_train_indices)
		remainder = train_count % fold_count
		if (remainder != 0):
			print(
				f"Warning - The number of samples <{train_count}> in your training Split\n" \
				f"is not evenly divisible by the `fold_count` <{fold_count}> you specified.\n" \
				f"This can result in misleading performance metrics for the last Fold.\n"
			)

		foldset = Foldset.create(
			fold_count = fold_count
			, random_state = random_state
			, bin_count = bin_count
			, splitset = splitset
		)
		try:
			# Stratified vs Unstratified.
			if (stratify_arr is None):
				# Nothing to stratify with.
				kf = KFold(
					n_splits=fold_count
					, shuffle=True
					, random_state=random_state
				)
				splitz_gen = kf.split(arr_train_indices)
			elif (stratify_arr is not None):
				skf = StratifiedKFold(
					n_splits=fold_count
					, shuffle=True
					, random_state=random_state
				)
				splitz_gen = skf.split(arr_train_indices, stratify_arr)

			i = -1
			for index_folds_train, index_fold_validation in splitz_gen:
				i+=1
				fold_samples = {}
				
				fold_samples["folds_train_combined"] = index_folds_train.tolist()
				fold_samples["fold_validation"] = index_fold_validation.tolist()

				Fold.create(
					fold_index = i
					, samples = fold_samples 
					, foldset = foldset
				)
		except:
			foldset.delete_instance() # Orphaned.
			raise
		return foldset


class Fold(BaseModel):
	"""
	- A Fold is 1 of many cross-validation sets generated as part of a Foldset.
	- The `samples` attribute contains the indices of `folds_train_combined` and `fold_validation`, 
	  where `fold_validation` is the rotating fold that gets left out.
	"""
	fold_index = IntegerField() # order within the Foldset.
	samples = JSONField()
	# contains_all_classes = BooleanField()
	
	foldset = ForeignKeyField(Foldset, backref='folds')