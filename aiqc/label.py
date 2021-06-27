from textwrap import dedent
from peewee import IntegerField, ForeignKeyField
from playhouse.sqlite_ext import JSONField
import numpy as np
from .basemodel import BaseModel
from .dataset import Dataset
from .utility import listify
from .labelcoder import Labelcoder

class Label(BaseModel):
	"""
	- Label accepts multiple columns in case it is already OneHotEncoded (e.g. tensors).
	- At this point, we assume that the Label is always a tabular dataset.
	"""
	columns = JSONField()
	column_count = IntegerField()
	unique_classes = JSONField(null=True) # For categoricals and binaries. None for continuous.
	
	dataset = ForeignKeyField(Dataset, backref='labels')
	
	def from_dataset(dataset_id:int, columns:list):
		d = Dataset.get_by_id(dataset_id)
		columns = listify(columns)

		if (d.dataset_type != 'tabular' and d.dataset_type != 'text'):
			raise ValueError(dedent(f"""
			Yikes - Labels can only be created from `dataset_type='tabular' or 'text'`.
			But you provided `dataset_type`: <{d.dataset_type}>
			"""))
		
		d_cols = Dataset.get_main_tabular(dataset_id).columns

		# Check that the user-provided columns exist.
		all_cols_found = all(col in d_cols for col in columns)
		if not all_cols_found:
			raise ValueError("\nYikes - You specified `columns` that do not exist in the Dataset.\n")

		# Check for duplicates of this label that already exist.
		cols_aplha = sorted(columns)
		d_labels = d.labels
		count = d_labels.count()
		if (count > 0):
			for l in d_labels:
				l_id = str(l.id)
				l_cols = l.columns
				l_cols_alpha = sorted(l_cols)
				if (cols_aplha == l_cols_alpha):
					raise ValueError(f"\nYikes - This Dataset already has Label <id:{l_id}> with the same columns.\nCannot create duplicate.\n")

		column_count = len(columns)

		label_df = Dataset.to_pandas(id=dataset_id, columns=columns)
		"""
		- When multiple columns are provided, they must be OHE.
		- Figure out column count because classification_binary and associated 
		metrics can't be run on > 2 columns.
		- Negative values do not alter type of numpy int64 and float64 arrays.
		"""
		if (column_count > 1):
			unique_values = []
			for c in columns:
				uniques = label_df[c].unique()
				unique_values.append(uniques)
				if (len(uniques) == 1):
					print(
						f"Warning - There is only 1 unique value for this label column.\n" \
						f"Unique value: <{uniques[0]}>\n" \
						f"Label column: <{c}>\n"
					)
			flat_uniques = np.concatenate(unique_values).ravel()
			all_uniques = np.unique(flat_uniques).tolist()

			for i in all_uniques:
				if (
					((i == 0) or (i == 1)) 
					or 
					((i == 0.) or (i == 1.))
				):
					pass
				else:
					raise ValueError(dedent(f"""
					Yikes - When multiple columns are provided, they must be One Hot Encoded:
					Unique values of your columns were neither (0,1) or (0.,1.) or (0.0,1.0).
					The columns you provided contained these unique values: {all_uniques}
					"""))
			unique_classes = all_uniques
			
			del label_df
			# Now check if each row in the labels is truly OHE.
			label_arr = Dataset.to_numpy(id=dataset_id, columns=columns)
			for i, arr in enumerate(label_arr):
				if 1 in arr:
					arr = list(arr)
					arr.remove(1)
					if 1 in arr:
						raise ValueError(dedent(f"""
						Yikes - Label row <{i}> is supposed to be an OHE row,
						but it contains multiple hot columns where value is 1.
						"""))
				else:
					raise ValueError(dedent(f"""
					Yikes - Label row <{i}> is supposed to be an OHE row,
					but it contains no hot columns where value is 1.
					"""))

		elif (column_count == 1):
			# At this point, `label_df` is a single column df that needs to fected as a Series.
			col = columns[0]
			label_series = label_df[col]
			label_dtype = label_series.dtype
			if (np.issubdtype(label_dtype, np.floating)):
				unique_classes = None
			else:
				unique_classes = label_series.unique().tolist()
				class_count = len(unique_classes)

				if (
					(np.issubdtype(label_dtype, np.signedinteger))
					or
					(np.issubdtype(label_dtype, np.unsignedinteger))
				):
					if (class_count >= 5):
						print(
							f"Tip - Detected  `unique_classes >= {class_count}` for an integer Label." \
							f"If this Label is not meant to be categorical, then we recommend you convert to a float-based dtype." \
							f"Although you'll still be able to bin these integers when it comes time to make a Splitset."
						)
				if (class_count == 1):
					print(
						f"Tip - Only detected 1 unique label class. Should have 2 or more unique classes." \
						f"Your Label's only class was: <{unique_classes[0]}>."
					)

		l = Label.create(
			dataset = d
			, columns = columns
			, column_count = column_count
			, unique_classes = unique_classes
		)
		return l


	def to_pandas(id:int, samples:list=None):
		samples = listify(samples)
		l_frame = Label.get_label(id=id, numpy_or_pandas='pandas', samples=samples)
		return l_frame


	def to_numpy(id:int, samples:list=None):
		samples = listify(samples)
		l_arr = Label.get_label(id=id, numpy_or_pandas='numpy', samples=samples)
		return l_arr


	def get_label(id:int, numpy_or_pandas:str, samples:list=None):
		samples = listify(samples)
		l = Label.get_by_id(id)
		l_cols = l.columns
		dataset_id = l.dataset.id

		if (numpy_or_pandas == 'numpy'):
			lf = Dataset.to_numpy(
				id = dataset_id
				, columns = l_cols
				, samples = samples
			)
		elif (numpy_or_pandas == 'pandas'):
			lf = Dataset.to_pandas(
				id = dataset_id
				, columns = l_cols
				, samples = samples
			)
		return lf


	def get_dtypes(
		id:int
	):
		l = Label.get_by_id(id)

		dataset = l.dataset
		l_cols = l.columns
		tabular_dtype = Dataset.get_main_tabular(dataset.id).dtypes

		label_dtypes = {}
		for key,value in tabular_dtype.items():
			for col in l_cols:         
				if (col == key):
					label_dtypes[col] = value
					# Exit `col` loop early becuase matching `col` found.
					break
		return label_dtypes


	def make_labelcoder(
		id:int
		, sklearn_preprocess:object
	):
		lc = Labelcoder.from_label(
			label_id = id
			, sklearn_preprocess = sklearn_preprocess
		)
		return lc


	def get_latest_labelcoder(id:int):
		label = Label.get_by_id(id)
		labelcoders = list(label.labelcoders)
		# Check if list empty.
		if (not labelcoders):
			return None
		else:
			return labelcoders[-1]