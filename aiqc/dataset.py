import os, tqdm, requests, validators
from textwrap import dedent
from natsort import natsorted 
from peewee import CharField, IntegerField
from sklearn.feature_extraction.text import CountVectorizer
from .label import Label
from .feature import Feature
from .file import File
from .basemodel import BaseModel
from .utility import listify
from PIL import Image as Imaje
import numpy as np
import pandas as pd

class Dataset(BaseModel):
	"""
	The sub-classes are not 1-1 tables. They simply provide namespacing for functions
	to avoid functions riddled with if statements about dataset_type and null parameters.
	"""
	dataset_type = CharField() #tabular, image, sequence, graph, audio.
	file_count = IntegerField() # only includes file_types that match the dataset_type.
	source_path = CharField(null=True)


	def make_label(id:int, columns:list):
		columns = listify(columns)
		l = Label.from_dataset(dataset_id=id, columns=columns)
		return l


	def make_feature(
		id:int
		, include_columns:list = None
		, exclude_columns:list = None
	):
		include_columns = listify(include_columns)
		exclude_columns = listify(exclude_columns)
		feature = Feature.from_dataset(
			dataset_id = id
			, include_columns = include_columns
			, exclude_columns = exclude_columns
		)
		return feature


	def to_pandas(id:int, columns:list=None, samples:list=None):
		dataset = Dataset.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular'):
			df = Dataset.Tabular.to_pandas(id=dataset.id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'text'):
			df = Dataset.Text.to_pandas(id=dataset.id, columns=columns, samples=samples)
		elif ((dataset.dataset_type == 'image') or (dataset.dataset_type == 'sequence')):
			raise ValueError("\nYikes - `dataset_type={dataset.dataset_type}` does not have a `to_pandas()` method.\n")
		return df


	def to_numpy(id:int, columns:list=None, samples:list=None):
		dataset = Dataset.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular'):
			arr = Dataset.Tabular.to_numpy(id=id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'image'):
			if (columns is not None):
				raise ValueError("\nYikes - `Dataset.Image.to_numpy` does not accept a `columns` argument.\n")
			arr = Dataset.Image.to_numpy(id=id, samples=samples)
		elif (dataset.dataset_type == 'text'):
			arr = Dataset.Text.to_numpy(id=id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'sequence'):
			arr = Dataset.Sequence.to_numpy(id=id, columns=columns, samples=samples)
		return arr


	def to_strings(id:int, samples:list=None):	
		dataset = Dataset.get_by_id(id)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular' or dataset.dataset_type == 'image'):
			raise ValueError("\nYikes - This Dataset class does not have a `to_strings()` method.\n")
		elif (dataset.dataset_type == 'text'):
			return Dataset.Text.to_strings(id=dataset.id, samples=samples)


	def sorted_file_list(dir_path:str):
		if (not os.path.exists(dir_path)):
			raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(dir_path)`:\n{dir_path}\n")
		path = os.path.abspath(dir_path)
		if (os.path.isdir(path) == False):
			raise ValueError(f"\nYikes - The path that you provided is not a directory:{path}\n")
		file_paths = os.listdir(path)
		# prune hidden files and directories.
		file_paths = [f for f in file_paths if not f.startswith('.')]
		file_paths = [f for f in file_paths if not os.path.isdir(f)]
		if not file_paths:
			raise ValueError(f"\nYikes - The directory that you provided has no files in it:{path}\n")
		# folder path is already absolute
		file_paths = [os.path.join(path, f) for f in file_paths]
		file_paths = natsorted(file_paths)
		return file_paths


	def get_main_file(id:int):
		dataset = Dataset.get_by_id(id)

		if (dataset.dataset_type == 'image'):
			raise ValueError("\n Dataset class does not support get_main_file() method for `image` data type,\n")

		file = File.select().join(Dataset).where(
			Dataset.id==id, File.file_type=='tabular', File.file_index==0
		)[0]
		return file

	def get_main_tabular(id:int):
		"""
		Works on both `Dataset.Tabular`, `Dataset.Sequence`, and `Dataset.Text`
		"""
		file = Dataset.get_main_file(id)
		return file.tabulars[0]


	def arr_validate(ndarray):
		if (type(ndarray).__name__ != 'ndarray'):
			raise ValueError("\nYikes - The `ndarray` you provided is not of the type 'ndarray'.\n")
		if (ndarray.dtype.names is not None):
			raise ValueError(dedent("""
			Yikes - Sorry, we do not support NumPy Structured Arrays.
			However, you can use the `dtype` dict and `column_names` to handle each column specifically.
			"""))
		if (ndarray.size == 0):
			raise ValueError("\nYikes - The ndarray you provided is empty: `ndarray.size == 0`.\n")



	class Tabular():
		"""
		- Does not inherit the Dataset class e.g. `class Tabular(Dataset):`
		  because then ORM would make a separate table for it.
		- It is just a collection of methods and default variables.
		"""
		dataset_type = 'tabular'
		file_index = 0
		file_count = 1

		def from_path(
			file_path:str
			, source_file_format:str
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, skip_header_rows:object = 'infer'
			, ingest:bool = True
		):
			column_names = listify(column_names)

			accepted_formats = ['csv', 'tsv', 'parquet']
			if (source_file_format not in accepted_formats):
				raise ValueError(f"\nYikes - Available file formats include csv, tsv, and parquet.\nYour file format: {source_file_format}\n")

			if (not os.path.exists(file_path)):
				raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(file_path)`:\n{file_path}\n")

			if (not os.path.isfile(file_path)):
				raise ValueError(dedent(
					f"Yikes - The path you provided is a directory according to `os.path.isfile(file_path)`:" \
					f"{file_path}" \
					f"But `dataset_type=='tabular'` only supports a single file, not an entire directory.`"
				))

			# Use the raw, not absolute path for the name.
			if (name is None):
				name = file_path

			source_path = os.path.abspath(file_path)

			dataset = Dataset.create(
				dataset_type = Dataset.Tabular.dataset_type
				, file_count = Dataset.Tabular.file_count
				, source_path = source_path
				, name = name
			)

			try:
				File.Tabular.from_file(
					path = file_path
					, source_file_format = source_file_format
					, dtype = dtype
					, column_names = column_names
					, skip_header_rows = skip_header_rows
					, ingest = ingest
					, dataset_id = dataset.id
				)
			except:
				dataset.delete_instance() # Orphaned.
				raise

			return dataset

		
		def from_pandas(
			dataframe:object
			, name:str = None
			, dtype:object = None
			, column_names:list = None
		):
			column_names = listify(column_names)

			if (type(dataframe).__name__ != 'DataFrame'):
				raise ValueError("\nYikes - The `dataframe` you provided is not `type(dataframe).__name__ == 'DataFrame'`\n")

			dataset = Dataset.create(
				file_count = Dataset.Tabular.file_count
				, dataset_type = Dataset.Tabular.dataset_type
				, name = name
				, source_path = None
			)

			try:
				File.Tabular.from_pandas(
					dataframe = dataframe
					, dtype = dtype
					, column_names = column_names
					, dataset_id = dataset.id
				)
			except:
				dataset.delete_instance() # Orphaned.
				raise 
			return dataset


		def from_numpy(
			ndarray:object
			, name:str = None
			, dtype:object = None
			, column_names:list = None
		):
			column_names = listify(column_names)
			Dataset.arr_validate(ndarray)

			dimensions = len(ndarray.shape)
			if (dimensions > 2) or (dimensions < 1):
				raise ValueError(dedent(f"""
				Yikes - Tabular Datasets only support 1D and 2D arrays.
				Your array dimensions had <{dimensions}> dimensions.
				"""))
			
			dataset = Dataset.create(
				file_count = Dataset.Tabular.file_count
				, name = name
				, source_path = None
				, dataset_type = Dataset.Tabular.dataset_type
			)
			try:
				File.Tabular.from_numpy(
					ndarray = ndarray
					, dtype = dtype
					, column_names = column_names
					, dataset_id = dataset.id
				)
			except:
				dataset.delete_instance() # Orphaned.
				raise 
			return dataset


		def to_pandas(
			id:int
			, columns:list = None
			, samples:list = None
		):
			file = Dataset.get_main_file(id)#`id` belongs to dataset, not file
			columns = listify(columns)
			samples = listify(samples)
			df = File.Tabular.to_pandas(id=file.id, samples=samples, columns=columns)
			return df


		def to_numpy(
			id:int
			, columns:list = None
			, samples:list = None
		):
			dataset = Dataset.get_by_id(id)
			columns = listify(columns)
			samples = listify(samples)
			# This calls the method above. It does not need `.Tabular`
			df = dataset.to_pandas(columns=columns, samples=samples)
			ndarray = df.to_numpy()
			return ndarray

	
	class Image():
		dataset_type = 'image'

		def from_folder(
			folder_path:str
			, name:str = None
			, pillow_save:dict = {}
			, ingest:bool = True
		):
			if ((pillow_save!={}) and (ingest==False)):
				raise ValueError("\nYikes - `pillow_save` cannot be defined if `ingest==False`.\n")
			if (name is None):
				name = folder_path
			source_path = os.path.abspath(folder_path)

			file_paths = Dataset.sorted_file_list(source_path)
			file_count = len(file_paths)

			dataset = Dataset.create(
				file_count = file_count
				, name = name
				, source_path = source_path
				, dataset_type = Dataset.Image.dataset_type
			)
			
			#Make sure the shape and mode of each image are the same before writing the Dataset.
			sizes = []
			modes = []
			
			for i, path in enumerate(tqdm(
				file_paths
				, desc = "🖼️ Validating Images 🖼️"
				, ncols = 85
			)):
				img = Imaje.open(path)
				sizes.append(img.size)
				modes.append(img.mode)

			if (len(set(sizes)) > 1):
				dataset.delete_instance()# Orphaned.
				raise ValueError(dedent(f"""
				Yikes - All images in the Dataset must be of the same width and height.
				`PIL.Image.size`\nHere are the unique sizes you provided:\n{set(sizes)}
				"""))
			elif (len(set(modes)) > 1):
				dataset.delete_instance()# Orphaned.
				raise ValueError(dedent(f"""
				Yikes - All images in the Dataset must be of the same mode aka colorscale.
				`PIL.Image.mode`\nHere are the unique modes you provided:\n{set(modes)}
				"""))

			try:
				for i, p in enumerate(tqdm(
					file_paths
					, desc = "🖼️ Ingesting Images 🖼️"
					, ncols = 85
				)):
					File.Image.from_file(
						path = p
						, pillow_save = pillow_save
						, file_index = i
						, ingest = ingest
						, dataset_id = dataset.id
					)
			except:
				dataset.delete_instance()
				raise
			return dataset


		def from_urls(
			urls:list
			, pillow_save:dict = {}
			, name:str = None
			, source_path:str = None
			, ingest:bool = True
		):
			if ((pillow_save!={}) and (ingest==False)):
				raise ValueError("\nYikes - `pillow_save` cannot be defined if `ingest==False`.\n")
			urls = listify(urls)
			for u in urls:
				validation = validators.url(u)
				if (validation != True): #`== False` doesn't work.
					raise ValueError(f"\nYikes - Invalid url detected within `urls` list:\n'{u}'\n")

			file_count = len(urls)

			dataset = Dataset.create(
				file_count = file_count
				, name = name
				, dataset_type = Dataset.Image.dataset_type
				, source_path = source_path
			)
			
			#Make sure the shape and mode of each image are the same before writing the Dataset.
			sizes = []
			modes = []
			
			for i, url in enumerate(tqdm(
					urls
					, desc = "🖼️ Validating Images 🖼️"
					, ncols = 85
			)):
				img = Imaje.open(
					requests.get(url, stream=True).raw
				)
				sizes.append(img.size)
				modes.append(img.mode)

			if (len(set(sizes)) > 1):
				raise ValueError(dedent(f"""
				Yikes - All images in the Dataset must be of the same width and height.
				`PIL.Image.size`\nHere are the unique sizes you provided:\n{set(sizes)}
				"""))
			elif (len(set(modes)) > 1):
				raise ValueError(dedent(f"""
				Yikes - All images in the Dataset must be of the same mode aka colorscale.
				`PIL.Image.mode`\nHere are the unique modes you provided:\n{set(modes)}
				"""))

			try:
				for i, url in enumerate(tqdm(
					urls
					, desc = "🖼️ Ingesting Images 🖼️"
					, ncols = 85
				)):
					File.Image.from_url(
						url = url
						, pillow_save = pillow_save
						, file_index = i
						, ingest = ingest
						, dataset_id = dataset.id
					)
				"""
				for i, url in enumerate(urls):  
					file = File.Image.from_url(
						url = url
						, pillow_save = pillow_save
						, file_index = i
						, dataset_id = dataset.id
					)
				"""
			except:
				dataset.delete_instance() # Orphaned.
				raise       
			return dataset


		def to_pillow(id:int, samples:list=None):
			"""
			- This does not have `columns` attrbute because it is only for fetching images.
			- Have to fetch as image before feeding into numpy `numpy.array(Image.open())`.
			- Future: could return the tabular data along with it.
			- Might need this for Preprocess where rotate images and such.
			"""
			samples = listify(samples)
			files = Dataset.Image.get_image_files(id, samples=samples)
			images = [f.Image.to_pillow(f.id) for f in files]
			return images


		def to_numpy(id:int, samples:list=None):
			"""
			- Because Pillow works directly with numpy, there's no need for pandas right now.
			- But downstream methods are using pandas.
			"""
			samples = listify(samples)
			images = Dataset.Image.to_pillow(id, samples=samples)
			images = [np.array(img) for img in images]
			images = np.array(images)
			"""
			- Pixel values range from 0-255.
			- `np.set_printoptions(threshold=99999)` to inspect for yourself.
			- It will look like some are all 0, but that's just the black edges.
			"""
			images = images/255
			return images


		def get_image_files(id:int, samples:list=None):
			samples = listify(samples)
			files = File.select().join(Dataset).where(
				Dataset.id==id, File.file_type=='image'
			).order_by(File.file_index)# Ascending by default.
			# Select from list by index.
			if (samples is not None):
				files = [files[i] for i in samples]
			return files


	class Text():
		dataset_type = 'text'
		file_count = 1
		column_name = 'TextData'

		def from_strings(
			strings: list,
			name: str = None
		):
			for expectedString in strings:
				if type(expectedString) !=  str:
					raise ValueError(f'\nThe input contains an object of type non-str type: {type(expectedString)}')

			dataframe = pd.DataFrame(strings, columns=[Dataset.Text.column_name], dtype="object")
			return Dataset.Text.from_pandas(dataframe, name)


		def from_pandas(
			dataframe:object,
			name:str = None, 
			dtype:object = None, 
			column_names:list = None
		):
			if Dataset.Text.column_name not in list(dataframe.columns):
				raise ValueError("\nYikes - The `dataframe` you provided doesn't contain 'TextData' column. Please rename the column containing text data to 'TextData'`\n")

			if dataframe[Dataset.Text.column_name].dtypes != 'O':
				raise ValueError("\nYikes - The `dataframe` you provided contains 'TextData' column with incorrect dtype: column dtype != object\n")

			dataset = Dataset.Tabular.from_pandas(dataframe, name, dtype, column_names)
			dataset.dataset_type = Dataset.Text.dataset_type
			dataset.save()
			return dataset


		def from_path(
			file_path:str
			, source_file_format:str
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, skip_header_rows:object = 'infer'
		):
			dataset = Dataset.Tabular.from_path(file_path, source_file_format, name, dtype, column_names, skip_header_rows)
			dataset.dataset_type = Dataset.Text.dataset_type
			dataset.save()
			return dataset


		def from_folder(
			folder_path:str, 
			name:str = None
		):
			if name is None:
				name = folder_path
			source_path = os.path.abspath(folder_path)
			
			input_files = Dataset.sorted_file_list(source_path)

			files_data = []
			for input_file in input_files:
				with open(input_file, 'r') as file_pointer:
					files_data.extend([file_pointer.read()])

			return Dataset.Text.from_strings(files_data, name)


		def to_pandas(
			id:int, 
			columns:list = None, 
			samples:list = None
		):
			df = Dataset.Tabular.to_pandas(id, columns, samples)

			if Dataset.Text.column_name not in columns:
				return df

			word_counts, feature_names = Dataset.Text.get_feature_matrix(df)
			df = pd.DataFrame(word_counts.todense(), columns = feature_names)
			return df

		
		def to_numpy(
			id:int, 
			columns:list = None, 
			samples:list = None
		):
			df = Dataset.Tabular.to_pandas(id, columns, samples)

			if Dataset.Text.column_name not in columns:
				return df.to_numpy()

			word_counts, feature_names = Dataset.Text.get_feature_matrix(df)
			return word_counts.todense()


		def get_feature_matrix(
			dataframe:object
		):
			count_vect = CountVectorizer(max_features = 200)
			word_counts = count_vect.fit_transform(dataframe[Dataset.Text.column_name].tolist())
			return word_counts, count_vect.get_feature_names()


		def to_strings(
			id:int, 
			samples:list = None
		):
			data_df = Dataset.Tabular.to_pandas(id, [Dataset.Text.column_name], samples)
			return data_df[Dataset.Text.column_name].tolist()


	class Sequence():
		dataset_type = 'sequence'

		def from_numpy(
			ndarray3D_or_npyPath:object
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, ingest:bool = True
		):
			if ((ingest==False) and (isinstance(dtype, dict))):
				raise ValueError("\nYikes - If `ingest==False` then `dtype` must be either a str or a single NumPy-based type.\n")
			# Fetch array from .npy if it is not an in-memory array.
			if (str(ndarray3D_or_npyPath.__class__) != "<class 'numpy.ndarray'>"):
				if (not isinstance(ndarray3D_or_npyPath, str)):
					raise ValueError("\nYikes - If `ndarray3D_or_npyPath` is not an array then it must be a string-based path.\n")
				if (not os.path.exists(ndarray3D_or_npyPath)):
					raise ValueError("\nYikes - The path you provided does not exist according to `os.path.exists(ndarray3D_or_npyPath)`\n")
				if (not os.path.isfile(ndarray3D_or_npyPath)):
					raise ValueError("\nYikes - The path you provided is not a file according to `os.path.isfile(ndarray3D_or_npyPath)`\n")
				source_path = ndarray3D_or_npyPath
				try:
					# `allow_pickle=False` prevented it from reading the file.
					ndarray_3D = np.load(file=ndarray3D_or_npyPath)
				except:
					print("\nYikes - Failed to `np.load(file=ndarray3D_or_npyPath)` with your `ndarray3D_or_npyPath`:\n")
					print(f"{ndarray3D_or_npyPath}\n")
					raise
			elif (str(ndarray3D_or_npyPath.__class__) == "<class 'numpy.ndarray'>"):
				source_path = None
				ndarray_3D = ndarray3D_or_npyPath 

			column_names = listify(column_names)
			Dataset.arr_validate(ndarray_3D)

			dimensions = len(ndarray_3D.shape)
			if (dimensions != 3):
				raise ValueError(dedent(f"""
				Yikes - Sequence Datasets can only be constructed from 3D arrays.
				Your array dimensions had <{dimensions}> dimensions.
				"""))

			file_count = len(ndarray_3D)
			dataset = Dataset.create(
				file_count = file_count
				, name = name
				, dataset_type = Dataset.Sequence.dataset_type
				, source_path = source_path
			)

			#Make sure the shape and mode of each image are the same before writing the Dataset.
			shapes = []
			for i, arr in enumerate(tqdm(
				ndarray_3D
				, desc = "⏱️ Validating Sequences 🧬"
				, ncols = 85
			)):
				shapes.append(arr.shape)

			if (len(set(shapes)) > 1):
				dataset.delete_instance()# Orphaned.
				raise ValueError(dedent(f"""
				Yikes - All 2D arrays in the Dataset must be of the shape.
				`ndarray.shape`\nHere are the unique sizes you provided:\n{set(shapes)}
				"""))

			try:
				for i, arr in enumerate(tqdm(
					ndarray_3D
					, desc = "⏱️ Ingesting Sequences 🧬"
					, ncols = 85
				)):
					File.Tabular.from_numpy(
						ndarray = arr
						, dataset_id = dataset.id
						, column_names = column_names
						, dtype = dtype
						, _file_index = i
						, ingest = ingest
					)
			except:
				dataset.delete_instance() # Orphaned.
				raise
			return dataset


		def to_numpy(
			id:int, 
			columns:list = None, 
			samples:list = None
		):
			dataset = Dataset.get_by_id(id)
			columns = listify(columns)
			samples = listify(samples)
			
			if (samples is None):
				files = dataset.files
			elif (samples is not None):
				# Here the 'sample' is the entire file. Whereas, in 2D 'sample==row'.
				# So run a query to get those files: `<<` means `in`.
				files = File.select().join(Dataset).where(
					Dataset.id==dataset.id, File.file_index<<samples
				)
			files = list(files)
			# Then call them with the column filter.
			# So don't pass `samples=samples` to the file.
			list_2D = [f.to_numpy(columns=columns) for f in files]
			arr_3D = np.array(list_2D)
			return arr_3D


	# Graph
	# handle nodes and edges as separate tabular types?
	# node_data is pretty much tabular sequence (varied length) data right down to the columns.
	# the only unique thing is an edge_data for each Graph file.
	# attach multiple file types to a file File(id=1).tabular, File(id=1).graph?