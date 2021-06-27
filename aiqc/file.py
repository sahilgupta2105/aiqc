import requests, io, os, fsspec
from textwrap import dedent
from peewee import ForeignKeyField, IntegerField, CharField, BooleanField, BlobField
from playhouse.fields import PickleField
from playhouse.sqlite_ext import JSONField
from .basemodel import BaseModel
from .dataset import Dataset
from .utility import listify, colIndices_from_colNames
from PIL import Image as Imaje
import pandas as pd
import numpy as np

class File(BaseModel):
	"""
	- Due to the fact that different types of Files have different attributes
	  (e.g. File.Tabular columns=JSON or File.Graph nodes=Blob, edges=Blob), 
	  I am making each file type its own subclass and 1-1 table. This approach 
	  allows for the creation of custom File types.
	- If `blob=None` then isn't persisted therefore fetch from source_path or s3_path.
	- Note that `dtype` does not require every column to be included as a key in the dictionary.
	"""
	file_type = CharField()
	file_format = CharField() # png, jpg, parquet.
	file_index = IntegerField() # image, sequence, graph.
	shape = JSONField()
	is_ingested = BooleanField()
	skip_header_rows = PickleField(null=True) #Image does not have.
	source_path = CharField(null=True) # when `from_numpy` or `from_pandas`.
	blob = BlobField(null=True) # when `is_ingested==False`.

	dataset = ForeignKeyField(Dataset, backref='files')
	
	"""
	Classes are much cleaner than a knot of if statements in every method,
	and `=None` for every parameter.
	"""

	def to_numpy(id:int, columns:list=None, samples:list=None):
		file = File.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)

		if (file.file_type == 'tabular'):
			arr = File.Tabular.to_numpy(id=id, columns=columns, samples=samples)
		elif (file.file_type == 'image'):
			arr = File.Image.to_numpy(id=id, columns=columns, samples=samples)
		return arr


	class Tabular():
		file_type = 'tabular'

		def from_pandas(
			dataframe:object
			, dataset_id:int
			, dtype:object = None # Accepts a single str for the entire df, but utlimate it gets saved as one dtype per column.
			, column_names:list = None
			, source_path:str = None # passed in via from_file, but not from_numpy.
			, ingest:bool = True # from_file() method overwrites this.
			, file_format:str = 'parquet' # from_file() method overwrites this.
			, skip_header_rows:int = 'infer'
			, _file_index:int = 0 # Dataset.Sequence overwrites this.
		):
			column_names = listify(column_names)
			File.Tabular.df_validate(dataframe, column_names)

			# We need this metadata whether ingested or not.
			dataframe, columns, shape, dtype = File.Tabular.df_set_metadata(
				dataframe=dataframe, column_names=column_names, dtype=dtype
			)

			if (ingest==True):
				blob = File.Tabular.df_to_compressed_parquet_bytes(dataframe)
			elif (ingest==False):
				blob = None

			dataset = Dataset.get_by_id(dataset_id)

			file = File.create(
				blob = blob
				, file_type = File.Tabular.file_type
				, file_format = file_format
				, file_index = _file_index
				, shape = shape
				, source_path = source_path
				, skip_header_rows = skip_header_rows
				, is_ingested = ingest
				, dataset = dataset
			)

			try:
				Tabular.create(
					columns = columns
					, dtypes = dtype
					, file_id = file.id
				)
			except:
				file.delete_instance() # Orphaned.
				raise 
			return file


		def from_numpy(
			ndarray:object
			, dataset_id:int
			, column_names:list = None
			, dtype:object = None #Or single string.
			, _file_index:int = 0
			, ingest:bool = True
		):
			column_names = listify(column_names)
			"""
			Only supporting homogenous arrays because structured arrays are a pain
			when it comes time to convert them to dataframes. It complained about
			setting an index, scalar types, and dimensionality... yikes.
			
			Homogenous arrays keep dtype in `arr.dtype==dtype('int64')`
			Structured arrays keep column names in `arr.dtype.names==('ID', 'Ring')`
			Per column dtypes dtypes from structured array <https://stackoverflow.com/a/65224410/5739514>
			"""
			Dataset.arr_validate(ndarray)
			"""
			column_names and dict-based dtype will be handled by our `from_pandas()` method.
			`pd.DataFrame` method only accepts a single dtype str, or infers if None.
			"""
			df = pd.DataFrame(data=ndarray)
			file = File.Tabular.from_pandas(
				dataframe = df
				, dataset_id = dataset_id
				, dtype = dtype
				# Setting `column_names` will not overwrite the first row of homogenous array:
				, column_names = column_names
				, _file_index = _file_index
				, ingest = ingest
			)
			return file


		def from_file(
			path:str
			, source_file_format:str
			, dataset_id:int
			, dtype:object = None
			, column_names:list = None
			, skip_header_rows:object = 'infer'
			, ingest:bool = True
		):
			column_names = listify(column_names)
			df = File.Tabular.path_to_df(
				path = path
				, source_file_format = source_file_format
				, column_names = column_names
				, skip_header_rows = skip_header_rows
			)

			file = File.Tabular.from_pandas(
				dataframe = df
				, dataset_id = dataset_id
				, dtype = dtype
				, column_names = None # See docstring above.
				, source_path = path
				, file_format = source_file_format
				, skip_header_rows = skip_header_rows
				, ingest = ingest
			)
			return file


		def to_pandas(
			id:int
			, columns:list = None
			, samples:list = None
		):
			"""
			This function could be optimized to read columns and rows selectively
			rather than dropping them after the fact.
			https://stackoverflow.com/questions/64050609/pyarrow-read-parquet-via-column-index-or-order
			"""
			file = File.get_by_id(id)
			columns = listify(columns)
			samples = listify(samples)


			if (file.is_ingested==False):
				# future: check if `query_fetcher` defined.
				df = File.Tabular.path_to_df(
					path = file.source_path
					, source_file_format = file.file_format
					, column_names = columns
					, skip_header_rows = file.skip_header_rows
				)
			elif (file.is_ingested==True):
				df = pd.read_parquet(
					io.BytesIO(file.blob)
					, columns=columns
				)
			# Ensures columns are rearranged to be in the correct order.
			if ((columns is not None) and (df.columns.to_list() != columns)):
				df = df.filter(columns)
			# Specific rows.
			if (samples is not None):
				df = df.iloc[samples]
			
			# Accepts dict{'column_name':'dtype_str'} or a single str.
			tab = file.tabulars[0]
			df_dtypes = tab.dtypes
			if (df_dtypes is not None):
				if (isinstance(df_dtypes, dict)):
					if (columns is None):
						columns = tab.columns
					# Prunes out the excluded columns from the dtype dict.
					df_dtype_cols = list(df_dtypes.keys())
					for col in df_dtype_cols:
						if (col not in columns):
							del df_dtypes[col]
				elif (isinstance(df_dtypes, str)):
					pass #dtype just gets applied as-is.
				df = df.astype(df_dtypes)

			return df


		def to_numpy(
			id:int
			, columns:list = None
			, samples:list = None
		):
			"""
			This is the only place where to_numpy() relies on to_pandas(). 
			It does so because pandas is good with the parquet and dtypes.
			"""
			columns = listify(columns)
			samples = listify(samples)
			file = File.get_by_id(id)
			# Handles when Dataset.Sequence is stored as a single .npy file
			if ((file.dataset.dataset_type=='sequence') and (file.is_ingested==False)):
				# Subsetting a File via `samples` is irrelevant here because the entire File is 1 sample.
				# Subset the columns:
				if (columns is not None):
					col_indices = colIndices_from_colNames(
						column_names = file.tabulars[0].columns
						, desired_cols = columns
					)
				dtype = list(file.tabulars[0].dtypes.values())[0] #`ingest==False` only allows singular dtype.
				# Verified that it is lazy via `sys.getsizeof()`				
				lazy_load = np.load(file.dataset.source_path)
				if (columns is not None):
					# First accessor[] gets the 2D. Second accessor[] gets the 2D.
					arr = lazy_load[file.file_index][:,col_indices].astype(dtype)
				else:
					arr = lazy_load[file.file_index].astype(dtype)
			else:
				df = File.Tabular.to_pandas(id=id, columns=columns, samples=samples)
				arr = df.to_numpy()
			return arr

		#Future: Add to_tensor and from_tensor? Or will numpy suffice?  

		def pandas_stringify_columns(df, columns):
			"""
			- `columns` is user-defined.
			- Pandas will assign a range of int-based columns if there are no column names.
			  So I want to coerce them to strings because I don't want both string and int-based 
			  column names for when calling columns programmatically, 
			  and more importantly, 'ValueError: parquet must have string column names'
			"""
			cols_raw = df.columns.to_list()
			if (columns is None):
				# in case the columns were a range of ints.
				cols_str = [str(c) for c in cols_raw]
			else:
				cols_str = columns
			# dict from 2 lists
			cols_dct = dict(zip(cols_raw, cols_str))
			
			df = df.rename(columns=cols_dct)
			columns = df.columns.to_list()
			return df, columns


		def df_validate(dataframe:object, column_names:list):
			if (dataframe.empty):
				raise ValueError("\nYikes - The dataframe you provided is empty according to `df.empty`\n")

			if (column_names is not None):
				col_count = len(column_names)
				structure_col_count = dataframe.shape[1]
				if (col_count != structure_col_count):
					raise ValueError(dedent(f"""
					Yikes - The dataframe you provided has <{structure_col_count}> columns,
					but you provided <{col_count}> columns.
					"""))


		def df_set_metadata(
			dataframe:object
			, column_names:list = None
			, dtype:object = None
		):
			shape = {}
			shape['rows'], shape['columns'] = dataframe.shape[0], dataframe.shape[1]

			"""
			- Passes in user-defined columns in case they are specified.
			- Pandas auto-assigns int-based columns return a range when `df.columns`, 
			  but this forces each column name to be its own str.
			 """
			dataframe, columns = File.Tabular.pandas_stringify_columns(df=dataframe, columns=column_names)

			"""
			- At this point, user-provided `dtype` can be either a dict or a singular string/ class.
			- But a Pandas dataframe in-memory only has `dtypes` dict not a singular `dtype` str.
			- So we will ensure that there is 1 dtype per column.
			"""
			if (dtype is not None):
				# Accepts dict{'column_name':'dtype_str'} or a single str.
				try:
					dataframe = dataframe.astype(dtype)
				except:
					raise ValueError("\nYikes - Failed to apply the dtypes you specified to the data you provided.\n")
				"""
				Check if any user-provided dtype against actual dataframe dtypes to see if conversions failed.
				Pandas dtype seems robust in comparing dtypes: 
				Even things like `'double' == dataframe['col_name'].dtype` will pass when `.dtype==np.float64`.
				Despite looking complex, category dtype converts to simple 'category' string.
				"""
				if (not isinstance(dtype, dict)):
					# Inspect each column:dtype pair and check to see if it is the same as the user-provided dtype.
					actual_dtypes = dataframe.dtypes.to_dict()
					for col_name, typ in actual_dtypes.items():
						if (typ != dtype):
							raise ValueError(dedent(f"""
							Yikes - You specified `dtype={dtype},
							but Pandas did not convert it: `dataframe['{col_name}'].dtype == {typ}`.
							You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
							"""))
				elif (isinstance(dtype, dict)):
					for col_name, typ in dtype.items():
						if (typ != dataframe[col_name].dtype):
							raise ValueError(dedent(f"""
							Yikes - You specified `dataframe['{col_name}']:dtype('{typ}'),
							but Pandas did not convert it: `dataframe['{col_name}'].dtype == {dataframe[col_name].dtype}`.
							You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
							"""))
			"""
			Testing outlandish dtypes:
			- `DataFrame.to_parquet(engine='auto')` fails on:
			  'complex', 'longfloat', 'float128'.
			- `DataFrame.to_parquet(engine='auto')` succeeds on:
			  'string', np.uint8, np.double, 'bool'.
			
			- But the new 'string' dtype is not a numpy type!
			  so operations like `np.issubdtype` and `StringArray.unique().tolist()` fail.
			"""
			excluded_types = ['string', 'complex', 'longfloat', 'float128']
			actual_dtypes = dataframe.dtypes.to_dict().items()

			for col_name, typ in actual_dtypes:
				for et in excluded_types:
					if (et in str(typ)):
						raise ValueError(dedent(f"""
						Yikes - You specified `dtype['{col_name}']:'{typ}',
						but aiqc does not support the following dtypes: {excluded_types}
						"""))

			"""
			Now, we take the all of the resulting dataframe dtypes and save them.
			Regardless of whether or not they were user-provided.
			Convert the classed `dtype('float64')` to a string so we can use it in `.to_pandas()`
			"""
			dtype = {k: str(v) for k, v in actual_dtypes}
			
			# Each object has the potential to be transformed so each object must be returned.
			return dataframe, columns, shape, dtype


		def df_to_compressed_parquet_bytes(dataframe:object):
			"""
			- The Parquet file format naturally preserves pandas/numpy dtypes.
			  Originally, we were using the `pyarrow` engine, but it has poor timedelta dtype support.
			  https://towardsdatascience.com/stop-persisting-pandas-data-frames-in-csvs-f369a6440af5
			
			- Although `fastparquet` engine preserves timedelta dtype, but it does not work with BytesIO.
			  https://github.com/dask/fastparquet/issues/586#issuecomment-861634507
			"""
			fs = fsspec.filesystem("memory")
			temp_path = "memory://temp.parq"
			dataframe.to_parquet(
				temp_path
				, engine = "fastparquet"
				, compression = "gzip"
				, index = False
			)
			blob = fs.cat(temp_path)
			fs.delete(temp_path)
			return blob


		def path_to_df(
			path:str
			, source_file_format:str
			, column_names:list
			, skip_header_rows:object
		):
			"""
			Previously, I was using pyarrow for all tabular/ sequence file formats. 
			However, it had worse support for missing column names and header skipping.
			So I switched to pandas for handling csv/tsv, but read_parquet()
			doesn't let you change column names easily, so using pyarrow for parquet.
			""" 
			if (not os.path.exists(path)):
				raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(path)`:\n{path}\n")

			if (not os.path.isfile(path)):
				raise ValueError(f"\nYikes - The path you provided is not a file according to `os.path.isfile(path)`:\n{path}\n")

			if (source_file_format == 'tsv') or (source_file_format == 'csv'):
				if (source_file_format == 'tsv') or (source_file_format is None):
					sep='\t'
					source_file_format = 'tsv' # Null condition.
				elif (source_file_format == 'csv'):
					sep=','

				df = pd.read_csv(
					filepath_or_buffer = path
					, sep = sep
					, names = column_names
					, header = skip_header_rows
				)
			elif (source_file_format == 'parquet'):
				if (skip_header_rows != 'infer'):
					raise ValueError(dedent("""
					Yikes - The argument `skip_header_rows` is not supported for `source_file_format='parquet'`
					because Parquet stores column names as metadata.\n
					"""))
				df = pd.read_parquet(path=path, engine='fastparquet')
				df, columns = File.Tabular.pandas_stringify_columns(df=df, columns=column_names)
			return df


	class Image():
		file_type = 'image'

		def from_file(
			path:str
			, file_index:int
			, dataset_id:int
			, pillow_save:dict = {}
			, ingest:bool = True
		):
			if not os.path.exists(path):
				raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(path)`:\n{path}\n")
			if not os.path.isfile(path):
				raise ValueError(f"\nYikes - The path you provided is not a file according to `os.path.isfile(path)`:\n{path}\n")
			path = os.path.abspath(path)

			img = Imaje.open(path)
			shape = {
				'width': img.size[0]
				, 'height':img.size[1]
			}

			if (ingest==True):
				blob = io.BytesIO()
				img.save(blob, format=img.format, **pillow_save)
				blob = blob.getvalue()
			elif (ingest==False):
				blob = None

			dataset = Dataset.get_by_id(dataset_id)
			file = File.create(
				blob = blob
				, file_type = File.Image.file_type
				, file_format = img.format
				, file_index = file_index
				, shape = shape
				, source_path = path
				, is_ingested = ingest
				, dataset = dataset
			)
			try:
				Image.create(
					mode = img.mode
					, size = img.size
					, file = file
					, pillow_save = pillow_save
				)
			except:
				file.delete_instance() # Orphaned.
				raise
			return file


		def from_url(
			url:str
			, file_index:int
			, dataset_id:int
			, pillow_save:dict = {}
			, ingest:bool = True
		):
			# URL format is validated in `from_urls`.
			try:
				img = Imaje.open(
					requests.get(url, stream=True).raw
				)
			except:
				raise ValueError(f"\nYikes - Could not open file at this url with Pillow library:\n{url}\n")
			shape = {
				'width': img.size[0]
				, 'height':img.size[1]
			}

			if (ingest==True):
				blob = io.BytesIO()
				img.save(blob, format=img.format, **pillow_save)
				blob = blob.getvalue()
			elif (ingest==False):
				blob = None

			dataset = Dataset.get_by_id(dataset_id)
			file = File.create(
				blob = blob
				, file_type = File.Image.file_type
				, file_format = img.format
				, file_index = file_index
				, shape = shape
				, source_path = url
				, is_ingested = ingest
				, dataset = dataset
			)
			try:
				Image.create(
					mode = img.mode
					, size = img.size
					, file = file
					, pillow_save = pillow_save
				)
			except:
				file.delete_instance() # Orphaned.
				raise
			return file



		def to_pillow(id:int):
			#https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
			file = File.get_by_id(id)
			if (file.file_type != 'image'):
				raise ValueError(dedent(f"""
				Yikes - Only `file.file_type='image' can be converted to Pillow images.
				But you provided `file.file_type`: <{file.file_type}>
				"""))
			#`mode` must be 'r'": https://pillow.readthedocs.io/en/stable/reference/Image.html
			if (file.is_ingested==True):
				img_bytes = io.BytesIO(file.blob)
				img = Imaje.open(img_bytes, mode='r')
			elif (file.is_ingested==False):
				# Future: store `is_url`.
				try:
					img = Imaje.open(file.source_path, mode='r')
				except:
					img = Imaje.open(
						requests.get(file.source_path, stream=True).raw
						, mode='r'
					)
			return img



class Tabular(BaseModel):
	"""
	- Do not change `dtype=PickleField()` because we are stringifying the columns.
	  I was tempted to do so for types like `np.float`, but we parse the final
	  type that Pandas decides to use.
	"""
	# Is sequence just a subset of tabular with a file_index?
	columns = JSONField()
	dtypes = JSONField()

	file = ForeignKeyField(File, backref='tabulars')




class Image(BaseModel):
	#https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
	mode = CharField()
	size = PickleField()
	pillow_save = JSONField()

	file = ForeignKeyField(File, backref='images')