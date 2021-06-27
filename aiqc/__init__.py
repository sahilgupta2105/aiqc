import os, operator, multiprocessing, io, random, itertools, h5py, statistics, math, time, pprint, datetime
# Python utils.
from textwrap import dedent
# External utils.
from tqdm import tqdm #progress bar.
from natsort import natsorted #file sorting.
import appdirs #os-agonistic folder.
# ORM.
from peewee import CharField, IntegerField, BlobField, BooleanField, DateTimeField, ForeignKeyField
from playhouse.fields import PickleField
# ETL.
import pandas as pd
import numpy as np
# Preprocessing & metrics.
import sklearn

# Deep learning.
import keras
import torch
# Visualization.
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from .configuration import setup_database, destroy_database, get_db
from .basemodel import BaseModel
from .utility import listify, dill_serialize, dill_deserialize
from .dataset import Dataset
import constants
from .file import File, Tabular, Image
from .label import Label
from .feature import Feature, Featureset, Window
from .splitset import Splitset
from .foldset import Foldset, Fold
from .encoderset import Encoderset
from .labelcoder import Label, Labelcoder
from .featurecoder import Featurecoder
from playhouse.sqlite_ext import JSONField

name = "aiqc"
"""
https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
- 'fork' makes all variables on main process available to child process. OS attempts not to duplicate all variables.
- 'spawn' requires that variables be passed to child as args, and seems to play by pickle's rules (e.g. no func in func).

- In Python 3.8, macOS changed default from 'fork' to 'spawn' , which is how I learned all this.
- Windows does not support 'fork'. It supports 'spawn'. So basically I have to play by spawn/ pickle rules.
- Spawn/ pickle dictates (1) where execute_jobs func is placed, (2) if MetricsCutoff func works, (3) if tqdm output is visible.
- Update: now MetricsCutoff is not working in `fork` mode.
- Wrote the `poll_progress` func for 'spawn' situations.
- If everything hits the fan, `run_jobs(in_background=False)` for a normal for loop.
- Tried `concurrent.futures` but it only works with `.py` from command line.
"""
if (os.name != 'nt'):
	# If `force=False`, then `importlib.reload(aiqc)` triggers `RuntimeError: context already set`.
	multiprocessing.set_start_method('fork', force=True)

app_dir_no_trailing_slash = appdirs.user_data_dir("aiqc")
# Adds either a trailing slash or backslashes depending on OS.
app_dir = os.path.join(app_dir_no_trailing_slash, '')
default_config_path = app_dir + "config.json"
default_db_path = app_dir + "aiqc.sqlite3"

#==================================================
# CONFIGURATION
#==================================================
def setup():
	setup_database([	File, Tabular, Image,
			Dataset,
			Label, Feature, 
			Splitset, Featureset, Foldset, Fold, 
			Encoderset, Labelcoder, Featurecoder, 
			Algorithm, Hyperparamset, Hyperparamcombo,
			Queue, Jobset, Job, Predictor, Prediction,
			FittedEncoderset, FittedLabelcoder,
			Window
		])

def destroy_db(confirm:bool=False, rebuild:bool=False):
	destroy_database([	File, Tabular, Image,
			Dataset,
			Label, Feature, 
			Splitset, Featureset, Foldset, Fold, 
			Encoderset, Labelcoder, Featurecoder, 
			Algorithm, Hyperparamset, Hyperparamcombo,
			Queue, Jobset, Job, Predictor, Prediction,
			FittedEncoderset, FittedLabelcoder,
			Window
		], confirm, rebuild)



class Algorithm(BaseModel):
	"""
	- Remember, pytorch and mxnet handle optimizer/loss outside the model definition as part of the train.
	- Could do a `.py` file as an alternative to Pickle.

	- Currently waiting for coleifer to accept prospect of a DillField
	https://github.com/coleifer/peewee/issues/2385
	"""
	library = CharField()
	analysis_type = CharField()#classification_multi, classification_binary, regression, clustering.
	
	fn_build = BlobField()
	fn_lose = BlobField() # null? do unsupervised algs have loss?
	fn_optimize = BlobField()
	fn_train = BlobField()
	fn_predict = BlobField()


	# --- used by `select_fn_lose()` ---
	def keras_regression_lose(**hp):
		loser = keras.losses.MeanAbsoluteError()
		return loser
	
	def keras_binary_lose(**hp):
		loser = keras.losses.BinaryCrossentropy()
		return loser
	
	def keras_multiclass_lose(**hp):
		loser = keras.losses.CategoricalCrossentropy()
		return loser

	def pytorch_binary_lose(**hp):
		loser = torch.nn.BCELoss()
		return loser

	def pytorch_multiclass_lose(**hp):
		# ptrckblck says `nn.NLLLoss()` will work too.
		loser = torch.nn.CrossEntropyLoss()
		return loser

	def pytorch_regression_lose(**hp):
		loser = torch.nn.L1Loss()#mean absolute error.
		return loser

	# --- used by `select_fn_optimize()` ---
	"""
	- Eventually could help the user select an optimizer based on topology (e.g. depth),
	  but Adamax works great for me everywhere.
	 - `**hp` needs to be included because that's how it is called in training loop.
	"""
	def keras_optimize(**hp):
		optimizer = keras.optimizers.Adamax(learning_rate=0.01)
		return optimizer

	def pytorch_optimize(model, **hp):
		optimizer = torch.optim.Adamax(model.parameters(),lr=0.01)
		return optimizer

	# --- used by `select_fn_predict()` ---
	def keras_multiclass_predict(model, samples_predict):
		# Shows the probabilities of each class coming out of softmax neurons:
		# array([[9.9990356e-01, 9.6374511e-05, 3.3754202e-10],...])
		probabilities = model.predict(samples_predict['features'])
		# This is the official keras replacement for multiclass `.predict_classes()`
		# Returns one ordinal array per sample: `[[0][1][2][3]]` 
		prediction = np.argmax(probabilities, axis=-1)
		return prediction, probabilities

	def keras_binary_predict(model, samples_predict):
		# Sigmoid output is between 0 and 1.
		# It's not technically a probability, but it is still easy to interpret.
		probability = model.predict(samples_predict['features'])
		# This is the official keras replacement for binary classes `.predict_classes()`.
		# Returns one array per sample: `[[0][1][0][1]]`.
		prediction = (probability > 0.5).astype("int32")
		return prediction, probability

	def keras_regression_predict(model, samples_predict):
		prediction = model.predict(samples_predict['features'])
		# ^ Output is a single value, not `probability, prediction`
		return prediction

	def pytorch_binary_predict(model, samples_predict):
		probability = model(samples_predict['features'])
		# Convert tensor back to numpy for AIQC metrics.
		probability = probability.detach().numpy()
		prediction = (probability > 0.5).astype("int32")
		# Both objects are numpy.
		return prediction, probability

	def pytorch_multiclass_predict(model, samples_predict):
		probabilities = model(samples_predict['features'])
		# Convert tensor back to numpy for AIQC metrics.
		probabilities = probabilities.detach().numpy()
		prediction = np.argmax(probabilities, axis=-1)
		# Both objects are numpy.
		return prediction, probabilities

	def pytorch_regression_predict(model, samples_predict):
		prediction = model(samples_predict['features']).detach().numpy()
		return prediction


	def select_fn_lose(
		library:str,
		analysis_type:str
	):      
		fn_lose = None
		if (library == 'keras'):
			if (analysis_type == 'regression'):
				fn_lose = Algorithm.keras_regression_lose
			elif (analysis_type == 'classification_binary'):
				fn_lose = Algorithm.keras_binary_lose
			elif (analysis_type == 'classification_multi'):
				fn_lose = Algorithm.keras_multiclass_lose
		elif (library == 'pytorch'):
			if (analysis_type == 'regression'):
				fn_lose = Algorithm.pytorch_regression_lose
			elif (analysis_type == 'classification_binary'):
				fn_lose = Algorithm.pytorch_binary_lose
			elif (analysis_type == 'classification_multi'):
				fn_lose = Algorithm.pytorch_multiclass_lose
		# After each of the predefined approaches above, check if it is still undefined.
		if fn_lose is None:
			raise ValueError(dedent("""
			Yikes - You did not provide a `fn_lose`,
			and we don't have an automated function for your combination of 'library' and 'analysis_type'
			"""))
		return fn_lose

	def select_fn_optimize(library:str):
		fn_optimize = None
		if (library == 'keras'):
			fn_optimize = Algorithm.keras_optimize
		elif (library == 'pytorch'):
			fn_optimize = Algorithm.pytorch_optimize
		# After each of the predefined approaches above, check if it is still undefined.
		if (fn_optimize is None):
			raise ValueError(dedent("""
			Yikes - You did not provide a `fn_optimize`,
			and we don't have an automated function for your 'library'
			"""))
		return fn_optimize

	def select_fn_predict(
		library:str,
		analysis_type:str
	):
		fn_predict = None
		if (library == 'keras'):
			if (analysis_type == 'classification_multi'):
				fn_predict = Algorithm.keras_multiclass_predict
			elif (analysis_type == 'classification_binary'):
				fn_predict = Algorithm.keras_binary_predict
			elif (analysis_type == 'regression'):
				fn_predict = Algorithm.keras_regression_predict
		elif (library == 'pytorch'):
			if (analysis_type == 'classification_multi'):
				fn_predict = Algorithm.pytorch_multiclass_predict
			elif (analysis_type == 'classification_binary'):
				fn_predict = Algorithm.pytorch_binary_predict
			elif (analysis_type == 'regression'):
				fn_predict = Algorithm.pytorch_regression_predict

		# After each of the predefined approaches above, check if it is still undefined.
		if fn_predict is None:
			raise ValueError(dedent("""
			Yikes - You did not provide a `fn_predict`,
			and we don't have an automated function for your combination of 'library' and 'analysis_type'
			"""))
		return fn_predict


	def make(
		library:str
		, analysis_type:str
		, fn_build:object
		, fn_train:object
		, fn_predict:object = None
		, fn_lose:object = None
		, fn_optimize:object = None
		, description:str = None
	):
		library = library.lower()
		if ((library != 'keras') and (library != 'pytorch')):
			raise ValueError("\nYikes - Right now, the only libraries we support are 'keras' and 'pytorch'\nMore to come soon!\n")

		analysis_type = analysis_type.lower()
		supported_analyses = ['classification_multi', 'classification_binary', 'regression']
		if (analysis_type not in supported_analyses):
			raise ValueError(f"\nYikes - Right now, the only analytics we support are:\n{supported_analyses}\n")

		if (fn_predict is None):
			fn_predict = Algorithm.select_fn_predict(
				library=library, analysis_type=analysis_type
			)
		if (fn_optimize is None):
			fn_optimize = Algorithm.select_fn_optimize(library=library)
		if (fn_lose is None):
			fn_lose = Algorithm.select_fn_lose(
				library=library, analysis_type=analysis_type
			)

		funcs = [fn_build, fn_optimize, fn_train, fn_predict, fn_lose]
		for i, f in enumerate(funcs):
			is_func = callable(f)
			if (not is_func):
				raise ValueError(f"\nYikes - The following variable is not a function, it failed `callable(variable)==True`:\n\n{f}\n")

		fn_build = dill_serialize(fn_build)
		fn_optimize = dill_serialize(fn_optimize)
		fn_train = dill_serialize(fn_train)
		fn_predict = dill_serialize(fn_predict)
		fn_lose = dill_serialize(fn_lose)

		algorithm = Algorithm.create(
			library = library
			, analysis_type = analysis_type
			, fn_build = fn_build
			, fn_optimize = fn_optimize
			, fn_train = fn_train
			, fn_predict = fn_predict
			, fn_lose = fn_lose
			, description = description
		)
		return algorithm


	def make_hyperparamset(
		id:int
		, hyperparameters:dict
		, description:str = None
		, pick_count:int = None
		, pick_percent:float = None
	):
		hyperparamset = Hyperparamset.from_algorithm(
			algorithm_id = id
			, hyperparameters = hyperparameters
			, description = description
			, pick_count = pick_count
			, pick_percent = pick_percent
		)
		return hyperparamset


	def make_queue(
		id:int
		, splitset_id:int
		, repeat_count:int = 1
		, hyperparamset_id:int = None
		, foldset_id:int = None
		, hide_test:bool = False
	):
		queue = Queue.from_algorithm(
			algorithm_id = id
			, splitset_id = splitset_id
			, hyperparamset_id = hyperparamset_id
			, foldset_id = foldset_id
			, repeat_count = repeat_count
			, hide_test = hide_test
		)
		return queue



class Hyperparamset(BaseModel):
	"""
	- Not glomming this together with Algorithm and Preprocess because you can keep the Algorithm the same,
	  while running many different queues of hyperparams.
	- An algorithm does not have to have a hyperparamset. It can used fixed parameters.
	- `repeat_count` is the number of times to run a model, sometimes you just get stuck at local minimas.
	- `param_count` is the number of paramets that are being hypertuned.
	- `possible_combos_count` is the number of possible combinations of parameters.

	- On setting kwargs with `**` and a dict: https://stackoverflow.com/a/29028601/5739514
	"""
	description = CharField(null=True)
	hyperparamcombo_count = IntegerField()
	#strategy = CharField() # set to all by default #all/ random. this would generate a different dict with less params to try that should be persisted for transparency.

	hyperparameters = JSONField()

	algorithm = ForeignKeyField(Algorithm, backref='hyperparamsets')

	def from_algorithm(
		algorithm_id:int
		, hyperparameters:dict
		, description:str = None
		, pick_count:int = None
		, pick_percent:float = None
	):
		if ((pick_count is not None) and (pick_percent is not None)):
			raise ValueError("Yikes - Either `pick_count` or `pick_percent` can be provided, but not both.")

		algorithm = Algorithm.get_by_id(algorithm_id)

		# Construct the hyperparameter combinations
		params_names = list(hyperparameters.keys())
		params_lists = list(hyperparameters.values())

		# Make sure they are actually lists.
		for i, pl in enumerate(params_lists):
			params_lists[i] = listify(pl)

		# From multiple lists, come up with every unique combination.
		params_combos = list(itertools.product(*params_lists))
		hyperparamcombo_count = len(params_combos)

		params_combos_dicts = []
		# Dictionary comprehension for making a dict from two lists.
		for params in params_combos:
			params_combos_dict = {params_names[i]: params[i] for i in range(len(params_names))} 
			params_combos_dicts.append(params_combos_dict)
		
		# These are the random selection strategies.
		if (pick_count is not None):
			if (pick_count < 1):
				raise ValueError(f"\nYikes - pick_count:<{pick_count}> cannot be less than 1.\n")
			elif (pick_count > hyperparamcombo_count):
				print(f"\nInfo - pick_count:<{pick_count}> greater than the number of hyperparameter combinations:<{hyperparamcombo_count}>.\nProceeding with all combinations.\n")
			else:
				# `sample` handles replacement.
				params_combos_dicts = random.sample(params_combos_dicts, pick_count)
				hyperparamcombo_count = len(params_combos_dicts)
		elif (pick_percent is not None):
			if ((pick_percent > 1.0) or (pick_percent <= 0.0)):
				raise ValueError(f"\nYikes - pick_percent:<{pick_percent}> must be between 0.0 and 1.0.\n")
			else:
				select_count = math.ceil(hyperparamcombo_count * pick_percent)
				params_combos_dicts = random.sample(params_combos_dicts, select_count)
				hyperparamcombo_count = len(params_combos_dicts)

		# Now that we have the metadata about combinations
		hyperparamset = Hyperparamset.create(
			algorithm = algorithm
			, description = description
			, hyperparameters = hyperparameters
			, hyperparamcombo_count = hyperparamcombo_count
		)

		for i, c in enumerate(params_combos_dicts):
			Hyperparamcombo.create(
				combination_index = i
				, favorite = False
				, hyperparameters = c
				, hyperparamset = hyperparamset
			)
		return hyperparamset




class Hyperparamcombo(BaseModel):
	combination_index = IntegerField()
	favorite = BooleanField()
	hyperparameters = JSONField()

	hyperparamset = ForeignKeyField(Hyperparamset, backref='hyperparamcombos')


	def get_hyperparameters(id:int, as_pandas:bool=False):
		hyperparamcombo = Hyperparamcombo.get_by_id(id)
		hyperparameters = hyperparamcombo.hyperparameters
		
		params = []
		for k,v in hyperparameters.items():
			param = {"param":k, "value":v}
			params.append(param)
		
		if (as_pandas==True):
			df = pd.DataFrame.from_records(params, columns=['param','value'])
			return df
		elif (as_pandas==False):
			return hyperparameters




class Plot():
	"""
	Data is prepared in the Queue and Predictor classes
	before being fed into the methods below.
	"""

	def __init__(self):

		self.plot_template = dict(layout=go.Layout(
			font=dict(family='Avenir', color='#FAFAFA'),
			title=dict(x=0.05, y=0.95),
			titlefont=dict(family='Avenir'),
			plot_bgcolor='#181B1E',
			paper_bgcolor='#181B1E',
			hovermode='closest',
			hoverlabel=dict(
				bgcolor="#0F0F0F",
				font=dict(
					family="Avenir",
					size=15
				)
			)))

	def performance(self, dataframe:object):
		# The 2nd metric is the last 
		name_metric_2 = dataframe.columns.tolist()[-1]
		if (name_metric_2 == "accuracy"):
			display_metric_2 = "Accuracy"
		elif (name_metric_2 == "r2"):
			display_metric_2 = "R²"
		else:
			raise ValueError(dedent(f"""
			Yikes - The name of the 2nd metric to plot was neither 'accuracy' nor 'r2'.
			You provided: {name_metric_2}.
			The 2nd metric is supposed to be the last column of the dataframe provided.
			"""))

		fig = px.line(
			dataframe
			, title = 'Models Metrics by Split'
			, x = 'loss'
			, y = name_metric_2
			, color = 'predictor_id'
			, height = 600
			, hover_data = ['predictor_id', 'split', 'loss', name_metric_2]
			, line_shape='spline'
		)
		fig.update_traces(
			mode = 'markers+lines'
			, line = dict(width = 2)
			, marker = dict(
				size = 8
				, line = dict(
					width = 2
					, color = 'white'
				)
			)
		)
		fig.update_layout(
			xaxis_title = "Loss"
			, yaxis_title = display_metric_2
			, template = self.plot_template
		)
		fig.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.show()


	def learning_curve(self, dataframe:object, analysis_type:str, loss_skip_15pct:bool=False):
		"""Dataframe rows are epochs and columns are metric names."""

		# Spline seems to crash with too many points.
		if (dataframe.shape[0] >= 400):
			line_shape = 'linear'
		elif (dataframe.shape[0] < 400):
			line_shape = 'spline'

		df_loss = dataframe[['loss','val_loss']]
		df_loss = df_loss.rename(columns={"loss": "train_loss", "val_loss": "validation_loss"})
		df_loss = df_loss.round(3)

		if loss_skip_15pct:
			df_loss = df_loss.tail(round(df_loss.shape[0]*.85))

		fig_loss = px.line(
			df_loss
			, title = 'Training History: Loss'
			, line_shape = line_shape
		)
		fig_loss.update_layout(
			xaxis_title = "Epochs"
			, yaxis_title = "Loss"
			, legend_title = None
			, template = self.plot_template
			, height = 400
			, yaxis = dict(
				side = "right"
				, tickmode = 'auto'# When loss is initially high, the 0.1 tickmarks are overwhelming.
				, tick0 = -1
				, nticks = 9
			)
			, legend = dict(
				orientation="h"
				, yanchor="bottom"
				, y=1.02
				, xanchor="right"
				, x=1
			)
			, margin = dict(
				t = 5
				, b = 0
			),
		)
		fig_loss.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig_loss.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))

		if ("classification" in analysis_type):
			df_acc = dataframe[['accuracy', 'val_accuracy']]
			df_acc = df_acc.rename(columns={"accuracy": "train_accuracy", "val_accuracy": "validation_accuracy"})
			df_acc = df_acc.round(3)

			fig_acc = px.line(
			df_acc
				, title = 'Training History: Accuracy'
				, line_shape = line_shape
			)
			fig_acc.update_layout(
				xaxis_title = "Epochs"
				, yaxis_title = "accuracy"
				, legend_title = None
				, height = 400
				, template = self.plot_template
				, yaxis = dict(
				side = "right"
				, tickmode = 'linear'
				, tick0 = 0.0
				, dtick = 0.05
				)
				, legend = dict(
					orientation="h"
					, yanchor="bottom"
					, y=1.02
					, xanchor="right"
					, x=1
				)
				, margin = dict(
					t = 5
				),
			)
			fig_acc.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
			fig_acc.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
			fig_acc.show()
		fig_loss.show()

	def confusion_matrix(self, cm_by_split, labels):
		for split, cm in cm_by_split.items():
			# change each element of z to type string for annotations
			cm_text = [[str(y) for y in x] for x in cm]

			# set up figure
			fig = ff.create_annotated_heatmap(
				cm
				, x=labels
				, y=labels
				, annotation_text=cm_text
				, colorscale=px.colors.sequential.BuGn
				, showscale=True
				, colorbar={"title": 'Count'})

			# add custom xaxis title
			fig.add_annotation(dict(font=dict(color="white", size=12),
									x=0.5,
									y=1.2,
									showarrow=False,
									text="Predicted Label",
									xref="paper",
									yref="paper"))

			# add custom yaxis title
			fig.add_annotation(dict(font=dict(color="white", size=12),
									x=-0.4,
									y=0.5,
									showarrow=False,
									text="Actual Label",
									textangle=-90,
									xref="paper",
									yref="paper"))


			fig.update_layout(
				title=f"Confusion Matrix: {split.capitalize()}"
				, legend_title='Sample Count'
				, template=self.plot_template
				, height=375  # if too small, it won't render in Jupyter.
				, width=850
				, yaxis=dict(
					tickmode='linear'
					, tick0=0.0
					, dtick=1.0
					, tickfont = dict(
						size=10
					)
				)
				, xaxis=dict(
					categoryorder='category descending',
					 tickfont=dict(
						size=10
					)
				)
				, margin=dict(
					r=325
					, l=325
				)
			)

			fig.update_traces(hovertemplate =
							  """predicted: %{x}<br>actual: %{y}<br>count: %{z}<extra></extra>""")

			fig.show()


	def precision_recall(self, dataframe:object):
		fig = px.line(
			dataframe
			, x = 'recall'
			, y = 'precision'
			, color = 'split'
			, title = 'Precision-Recall Curves'
		)
		fig.update_layout(
			legend_title = None
			, template = self.plot_template
			, height = 500
			, yaxis = dict(
				side = "right"
				, tickmode = 'linear'
				, tick0 = 0.0
				, dtick = 0.05
			)
			, legend = dict(
				orientation="h"
				, yanchor="bottom"
				, y=1.02
				, xanchor="right"
				, x=1
			)
		)
		fig.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.show()


	def roc_curve(self, dataframe:object):
		fig = px.line(
			dataframe
			, x = 'fpr'
			, y = 'tpr'
			, color = 'split'
			, title = 'Receiver Operating Characteristic (ROC) Curves'
		)
		fig.update_layout(
			legend_title = None
			, template = self.plot_template
			, height = 500
			, xaxis = dict(
				title = "False Positive Rate (FPR)"
				, tick0 = 0.00
				, range = [-0.025,1]
			)
			, yaxis = dict(
				title = "True Positive Rate (TPR)"
				, side = "left"
				, tickmode = 'linear'
				, tick0 = 0.00
				, dtick = 0.05
				, range = [0,1.05]
			)
			, legend = dict(
				orientation="h"
				, yanchor="bottom"
				, y=1.02
				, xanchor="right"
				, x=1
			)
			, shapes=[
				dict(
					type = 'line'
					, y0=0, y1=1
					, x0=0, x1=1
					, line = dict(dash='dot', width=2, color='#3b4043')
			)]
		)
		fig.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.show()



class Queue(BaseModel):
	repeat_count = IntegerField()
	run_count = IntegerField()
	hide_test = BooleanField()

	algorithm = ForeignKeyField(Algorithm, backref='queues') 
	splitset = ForeignKeyField(Splitset, backref='queues')

	hyperparamset = ForeignKeyField(Hyperparamset, deferrable='INITIALLY DEFERRED', null=True, backref='queues')
	foldset = ForeignKeyField(Foldset, deferrable='INITIALLY DEFERRED', null=True, backref='queues')


	def from_algorithm(
		algorithm_id:int
		, splitset_id:int
		, repeat_count:int = 1
		, hide_test:bool=False
		, hyperparamset_id:int = None
		, foldset_id:int = None
	):
		algorithm = Algorithm.get_by_id(algorithm_id)
		library = algorithm.library
		splitset = Splitset.get_by_id(splitset_id)

		if (foldset_id is not None):
			foldset = Foldset.get_by_id(foldset_id)
		# Future: since unsupervised won't have a Label for flagging the analysis type, I am going to keep the `Algorithm.analysis_type` attribute for now.
		if (splitset.supervision == 'supervised'):
			# Validate combinations of alg.analysis_type, lbl.col_count, lbl.dtype, split/fold.bin_count
			analysis_type = algorithm.analysis_type
			label_col_count = splitset.label.column_count
			label_dtypes = list(splitset.label.get_dtypes().values())
			
			labelcoder = splitset.label.get_latest_labelcoder()

			if (labelcoder is not None):
				stringified_labelcoder = str(labelcoder.sklearn_preprocess)
			else:
				stringified_labelcoder = None

			if (label_col_count == 1):
				label_dtype = label_dtypes[0]

				if ('classification' in analysis_type): 
					if (np.issubdtype(label_dtype, np.floating)):
						raise ValueError("Yikes - Cannot have `Algorithm.analysis_type!='regression`, when Label dtype falls under `np.floating`.")

					if (labelcoder is not None):
						if (labelcoder.is_categorical == False):
							raise ValueError(dedent(f"""
								Yikes - `Algorithm.analysis_type=='classification_*'`, but 
								`Labelcoder.sklearn_preprocess={stringified_labelcoder}` was not found in known 'classification' encoders:
								{constants.categorical_encoders}
							"""))

						if ('_binary' in analysis_type):
							# Prevent OHE w classification_binary
							if (stringified_labelcoder.startswith("OneHotEncoder")):
								raise ValueError(dedent("""
								Yikes - `Algorithm.analysis_type=='classification_binary', but 
								`Labelcoder.sklearn_preprocess.startswith('OneHotEncoder')`.
								This would result in a multi-column output, but binary classification
								needs a single column output.
								Go back and make a Labelcoder with single column output preprocess like `Binarizer()` instead.
								"""))
						elif ('_multi' in analysis_type):
							if (library == 'pytorch'):
								# Prevent OHE w pytorch.
								if (stringified_labelcoder.startswith("OneHotEncoder")):
									raise ValueError(dedent("""
									Yikes - `(analysis_type=='classification_multi') and (library == 'pytorch')`, 
									but `Labelcoder.sklearn_preprocess.startswith('OneHotEncoder')`.
									This would result in a multi-column OHE output.
									However, neither `nn.CrossEntropyLoss` nor `nn.NLLLoss` support multi-column input.
									Go back and make a Labelcoder with single column output preprocess like `OrdinalEncoder()` instead.
									"""))
								elif (not stringified_labelcoder.startswith("OrdinalEncoder")):
									print(dedent("""
										Warning - When `(analysis_type=='classification_multi') and (library == 'pytorch')`
										We recommend you use `sklearn.preprocessing.OrdinalEncoder()` as a Labelcoder.
									"""))
							else:
								if (not stringified_labelcoder.startswith("OneHotEncoder")):
									print(dedent("""
										Warning - When performing non-PyTorch, multi-label classification on a single column,
										we recommend you use `sklearn.preprocessing.OneHotEncoder()` as a Labelcoder.
									"""))
					elif (
						(labelcoder is None) and ('_multi' in analysis_type) and (library != 'pytorch')
					):
						print(dedent("""
							Warning - When performing non-PyTorch, multi-label classification on a single column 
							without using a Labelcoder, Algorithm must have user-defined `fn_lose`, 
							`fn_optimize`, and `fn_predict`. We recommend you use 
							`sklearn.preprocessing.OneHotEncoder()` as a Labelcoder instead.
						"""))

					if (splitset.bin_count is not None):
						print(dedent("""
							Warning - `'classification' in Algorithm.analysis_type`, but `Splitset.bin_count is not None`.
							`bin_count` is meant for `Algorithm.analysis_type=='regression'`.
						"""))               
					if (foldset_id is not None):
						# Not doing an `and` because foldset can't be accessed if it doesn't exist.
						if (foldset.bin_count is not None):
							print(dedent("""
								Warning - `'classification' in Algorithm.analysis_type`, but `Foldset.bin_count is not None`.
								`bin_count` is meant for `Algorithm.analysis_type=='regression'`.
							"""))
				elif (analysis_type == 'regression'):
					if (labelcoder is not None):
						if (labelcoder.is_categorical == True):
							raise ValueError(dedent(f"""
								Yikes - `Algorithm.analysis_type=='regression'`, but 
								`Labelcoder.sklearn_preprocess={stringified_labelcoder}` was found in known categorical encoders:
								{constants.categorical_encoders}
							"""))

					if (
						(not np.issubdtype(label_dtype, np.floating))
						and
						(not np.issubdtype(label_dtype, np.unsignedinteger))
						and
						(not np.issubdtype(label_dtype, np.signedinteger))
					):
						raise ValueError("Yikes - `Algorithm.analysis_type == 'regression'`, but label dtype was neither `np.floating`, `np.unsignedinteger`, nor `np.signedinteger`.")
					
					if (splitset.bin_count is None):
						print("Warning - `Algorithm.analysis_type == 'regression'`, but `bin_count` was not set when creating Splitset.")                   
					if (foldset_id is not None):
						if (foldset.bin_count is None):
							print("Warning - `Algorithm.analysis_type == 'regression'`, but `bin_count` was not set when creating Foldset.")
							if (splitset.bin_count is not None):
								print("Warning - `bin_count` was set for Splitset, but not for Foldset. This leads to inconsistent stratification across samples.")
						elif (foldset.bin_count is not None):
							if (splitset.bin_count is None):
								print("Warning - `bin_count` was set for Foldset, but not for Splitset. This leads to inconsistent stratification across samples.")
				
			# We already know these are OHE based on Label creation, so skip dtype, bin, and encoder checks.
			elif (label_col_count > 1):
				if (analysis_type != 'classification_multi'):
					raise ValueError("Yikes - `Label.column_count > 1` but `Algorithm.analysis_type != 'classification_multi'`.")

		elif ((splitset.supervision != 'supervised') and (hide_test==True)):
			raise ValueError("\nYikes - Cannot have `hide_test==True` if `splitset.supervision != 'supervised'`.\n")

		if (foldset_id is not None):
			foldset =  Foldset.get_by_id(foldset_id)
			foldset_splitset = foldset.splitset
			if foldset_splitset != splitset:
				raise ValueError(f"\nYikes - The Foldset <id:{foldset_id}> and Splitset <id:{splitset_id}> you provided are not related.\n")
			folds = list(foldset.folds)
		else:
			# Just so we have an item to loop over as a null condition when creating Jobs.
			folds = [None]
			foldset = None

		if (hyperparamset_id is not None):
			hyperparamset = Hyperparamset.get_by_id(hyperparamset_id)
			combos = list(hyperparamset.hyperparamcombos)
		else:
			# Just so we have an item to loop over as a null condition when creating Jobs.
			combos = [None]
			hyperparamset = None

		# The null conditions set above (e.g. `[None]`) ensure multiplication by 1.
		run_count = len(combos) * len(folds) * repeat_count

		q = Queue.create(
			run_count = run_count
			, repeat_count = repeat_count
			, algorithm = algorithm
			, splitset = splitset
			, foldset = foldset
			, hyperparamset = hyperparamset
			, hide_test = hide_test
		)
 
		for c in combos:
			if (foldset is not None):
				jobset = Jobset.create(
					repeat_count = repeat_count
					, queue = q
					, hyperparamcombo = c
					, foldset = foldset
				)
			elif (foldset is None):
				jobset = None

			try:
				for f in folds:
					Job.create(
						queue = q
						, hyperparamcombo = c
						, fold = f
						, repeat_count = repeat_count
						, jobset = jobset
					)
			except:
				if (foldset is not None):
					jobset.delete_instance() # Orphaned.
					raise
		return q


	def poll_statuses(id:int, as_pandas:bool=False):
		queue = Queue.get_by_id(id)
		repeat_count = queue.repeat_count
		statuses = []
		for i in range(repeat_count):
			for j in queue.jobs:
				# Check if there is a Predictor with a matching repeat_index
				matching_predictor = Predictor.select().join(Job).join(Queue).where(
					Queue.id==queue.id, Job.id==j.id, Predictor.repeat_index==i
				)
				if (len(matching_predictor) == 1):
					r_id = matching_predictor[0].id
				elif (len(matching_predictor) == 0):
					r_id = None
				job_dct = {"job_id":j.id, "repeat_index":i, "predictor_id": r_id}
				statuses.append(job_dct)

		if (as_pandas==True):
			df = pd.DataFrame.from_records(statuses, columns=['job_id', 'repeat_index', 'predictor_id'])
			return df.round()
		elif (as_pandas==False):
			return statuses


	def poll_progress(id:int, raw:bool=False, loop:bool=False, loop_delay:int=3):
		"""
		- For background_process execution where progress bar not visible.
		- Could also be used for cloud jobs though.
		"""
		if (loop==False):
			statuses = Queue.poll_statuses(id)
			total = len(statuses)
			done_count = len([s for s in statuses if s['predictor_id'] is not None]) 
			percent_done = done_count / total

			if (raw==True):
				return percent_done
			elif (raw==False):
				done_pt05 = round(round(percent_done / 0.05) * 0.05, -int(math.floor(math.log10(0.05))))
				bars_filled = int(done_pt05 * 20)
				bars_blank = 20 - bars_filled
				meter = '|'
				for i in range(bars_filled):
					meter += '██'
				for i in range(bars_blank):
					meter += '--'
				meter += '|'
				print(f"🔮 Training Models 🔮 {meter} {done_count}/{total} : {int(percent_done*100)}%")
		elif (loop==True):
			while (loop==True):
				statuses = Queue.poll_statuses(id)
				total = len(statuses)
				done_count = len([s for s in statuses if s['predictor_id'] is not None]) 
				percent_done = done_count / total
				if (raw==True):
					return percent_done
				elif (raw==False):
					done_pt05 = round(round(percent_done / 0.05) * 0.05, -int(math.floor(math.log10(0.05))))
					bars_filled = int(done_pt05 * 20)
					bars_blank = 20 - bars_filled
					meter = '|'
					for i in range(bars_filled):
						meter += '██'
					for i in range(bars_blank):
						meter += '--'
					meter += '|'
					print(f"🔮 Training Models 🔮 {meter} {done_count}/{total} : {int(percent_done*100)}%", end='\r')
					#print()

				if (done_count == total):
					loop = False
					os.system("say Model training completed")
					break
				time.sleep(loop_delay)


	def run_jobs(id:int, in_background:bool=False, verbose:bool=False):
		queue = Queue.get_by_id(id)

		# Quick check to make sure all predictors aren't already complete.
		run_count = queue.run_count
		predictor_count = Predictor.select().join(Job).join(Queue).where(
			Queue.id == queue.id).count()
		if (run_count == predictor_count):
			print("\nAll Jobs have already completed.\n")
		else:
			if (run_count > predictor_count > 0):
				print("\nResuming Jobs...\n")
			job_statuses = Queue.poll_statuses(id)
			
			if (in_background==True):
				proc_name = "aiqc_queue_" + str(queue.id)
				proc_names = [p.name for p in multiprocessing.active_children()]
				if (proc_name in proc_names):
					raise ValueError(
						f"\nYikes - Cannot start this Queue because multiprocessing.Process.name '{proc_name}' is already running."
						f"\nIf need be, you can kill the existing Process with `queue.stop_jobs()`.\n"
					)
				
				# See notes at top of file about 'fork' vs 'spawn'
				proc = multiprocessing.Process(
					target = execute_jobs
					, name = proc_name
					, args = (job_statuses, verbose,) #Needs trailing comma.
				)
				proc.start()
				# proc terminates when `execute_jobs` finishes.
			elif (in_background==False):
				try:
					for j in tqdm(
						job_statuses
						, desc = "🔮 Training Models 🔮"
						, ncols = 100
					):
						if (j['predictor_id'] is None):
							Job.run(id=j['job_id'], verbose=verbose, repeat_index=j['repeat_index'])
				except (KeyboardInterrupt):
					# So that we don't get nasty error messages when interrupting a long running loop.
					print("\nQueue was gracefully interrupted.\n")


	def stop_jobs(id:int):
		# SQLite is ACID (D = Durable). If transaction is interrupted mid-write, then it is rolled back.
		queue = Queue.get_by_id(id)
		
		proc_name = f"aiqc_queue_{queue.id}"
		current_procs = [p.name for p in multiprocessing.active_children()]
		if (proc_name not in current_procs):
			raise ValueError(f"\nYikes - Cannot terminate `multiprocessing.Process.name` '{proc_name}' because it is not running.\n")

		processes = multiprocessing.active_children()
		for p in processes:
			if (p.name == proc_name):
				try:
					p.terminate()
				except:
					raise Exception(f"\nYikes - Failed to terminate `multiprocessing.Process` '{proc_name}.'\n")
				else:
					print(f"\nKilled `multiprocessing.Process` '{proc_name}' spawned from aiqc.Queue <id:{queue.id}>\n")


	def metrics_to_pandas(
		id:int
		, selected_metrics:list=None
		, sort_by:list=None
		, ascending:bool=False
	):
		queue = Queue.get_by_id(id)
		selected_metrics = listify(selected_metrics)
		sort_by = listify(sort_by)
		
		queue_predictions = Prediction.select().join(
			Predictor).join(Job).where(Job.queue==id
		).order_by(Prediction.id)
		queue_predictions = list(queue_predictions)

		if (not queue_predictions):
			print(dedent("""
				~:: Patience, young Padawan ::~

				Completed, your Jobs are not. So Predictors to be had, there are None.
			"""))
			return None

		metric_names = list(list(queue_predictions[0].metrics.values())[0].keys())#bad.
		if (selected_metrics is not None):
			for m in selected_metrics:
				if (m not in metric_names):
					raise ValueError(dedent(f"""
					Yikes - The metric '{m}' does not exist in `Predictor.metrics`.
					Note: the metrics available depend on the `Queue.analysis_type`.
					"""))
		elif (selected_metrics is None):
			selected_metrics = metric_names

		# Unpack the split data from each Predictor and tag it with relevant Queue metadata.
		split_metrics = []
		for prediction in queue_predictions:
			predictor = prediction.predictor
			for split_name,metrics in prediction.metrics.items():

				split_metric = {}
				if (predictor.job.hyperparamcombo is not None):
					split_metric['hyperparamcombo_id'] = predictor.job.hyperparamcombo.id
				elif (predictor.job.hyperparamcombo is None):
					split_metric['hyperparamcombo_id'] = None

				if (queue.foldset is not None):
					split_metric['jobset_id'] = predictor.job.jobset.id
					split_metric['fold_index'] = predictor.job.fold.fold_index
				split_metric['job_id'] = predictor.job.id
				if (predictor.job.repeat_count > 1):
					split_metric['repeat_index'] = predictor.repeat_index

				split_metric['predictor_id'] = prediction.id
				split_metric['split'] = split_name

				for metric_name,metric_value in metrics.items():
					# Check whitelist.
					if metric_name in selected_metrics:
						split_metric[metric_name] = metric_value

				split_metrics.append(split_metric)

		column_names = list(split_metrics[0].keys())
		if (sort_by is not None):
			for name in sort_by:
				if (name not in column_names):
					raise ValueError(f"\nYikes - Column '{name}' not found in metrics dataframe.\n")
			df = pd.DataFrame.from_records(split_metrics).sort_values(
				by=sort_by, ascending=ascending
			)
		elif (sort_by is None):
			df = pd.DataFrame.from_records(split_metrics).sort_values(
				by=['predictor_id'], ascending=ascending
			)
		return df


	def metrics_aggregate_to_pandas(
		id:int
		, ascending:bool=False
		, selected_metrics:list=None
		, selected_stats:list=None
		, sort_by:list=None
	):
		selected_metrics = listify(selected_metrics)
		selected_stats = listify(selected_stats)
		sort_by = listify(sort_by)

		queue_predictions = Prediction.select().join(
			Predictor).join(Job).where(Job.queue==id
		).order_by(Prediction.id)
		queue_predictions = list(queue_predictions)

		if (not queue_predictions):
			print("\n~:: Patience, young Padawan ::~\n\nThe Jobs have not completed yet, so there are no Predictors to be had.\n")
			return None

		metrics_aggregate = queue_predictions[0].metrics_aggregate
		metric_names = list(metrics_aggregate.keys())
		stat_names = list(list(metrics_aggregate.values())[0].keys())

		if (selected_metrics is not None):
			for m in selected_metrics:
				if (m not in metric_names):
					raise ValueError(dedent(f"""
					Yikes - The metric '{m}' does not exist in `Predictor.metrics_aggregate`.
					Note: the metrics available depend on the `Queue.analysis_type`.
					"""))
		elif (selected_metrics is None):
			selected_metrics = metric_names

		if (selected_stats is not None):
			for s in selected_stats:
				if (s not in stat_names):
					raise ValueError(f"\nYikes - The statistic '{s}' does not exist in `Predictor.metrics_aggregate`.\n")
		elif (selected_stats is None):
			selected_stats = stat_names

		predictions_stats = []
		for prediction in queue_predictions:
			predictor = prediction.predictor
			for metric, stats in prediction.metrics_aggregate.items():
				# Check whitelist.
				if (metric in selected_metrics):
					stats['metric'] = metric
					stats['predictor_id'] = prediction.id
					if (predictor.job.repeat_count > 1):
						stats['repeat_index'] = predictor.repeat_index
					if (predictor.job.fold is not None):
						stats['jobset_id'] = predictor.job.jobset.id
						stats['fold_index'] = predictor.job.fold.fold_index
					else:
						stats['job_id'] = predictor.job.id
					stats['hyperparamcombo_id'] = predictor.job.hyperparamcombo.id

					predictions_stats.append(stats)

		# Cannot edit dictionary while key-values are being accessed.
		for stat in stat_names:
			if (stat not in selected_stats):
				for s in predictions_stats:
					s.pop(stat)# Errors if not found.

		#Reverse the order of the dictionary keys.
		predictions_stats = [dict(reversed(list(d.items()))) for d in predictions_stats]
		column_names = list(predictions_stats[0].keys())

		if (sort_by is not None):
			for name in sort_by:
				if (name not in column_names):
					raise ValueError(f"\nYikes - Column '{name}' not found in aggregate metrics dataframe.\n")
			df = pd.DataFrame.from_records(predictions_stats).sort_values(
				by=sort_by, ascending=ascending
			)
		elif (sort_by is None):
			df = pd.DataFrame.from_records(predictions_stats)
		return df


	def plot_performance(
		id:int
		, max_loss:float=None
		, min_accuracy:float=None
		, min_r2:float=None
	):
		"""
		Originally I had `min_metric_2` not `min_accuracy` and `min_r2`,
		but that would be confusing for users, so I went with informative 
		erro messages instead.
		"""
		queue = Queue.get_by_id(id)
		analysis_type = queue.algorithm.analysis_type

		# Now we need to filter the df based on the specified criteria.
		if ("classification" in analysis_type):
			if (min_r2 is not None):
				raise ValueError("\nYikes - Cannot use argument `min_r2` if `'classification' in queue.analysis_type`.\n")
			if (min_accuracy is None):
				min_accuracy = 0.0
			min_metric_2 = min_accuracy
			name_metric_2 = "accuracy"
		elif (analysis_type == 'regression'):
			if (min_accuracy is not None):
				raise ValueError("\nYikes - Cannot use argument `min_accuracy` if `queue.analysis_type='regression'`.\n")
			if (min_r2 is None):
				min_r2 = -1.0
			min_metric_2 = min_r2
			name_metric_2 = "r2"

		if (max_loss is None):
			max_loss = float('inf')
			
		df = queue.metrics_to_pandas()
		if (df is None):
			# Warning message handled by `metrics_to_pandas() above`.
			return None
		qry_str = "(loss >= {}) | ({} <= {})".format(max_loss, name_metric_2, min_metric_2)
		failed = df.query(qry_str)
		failed_runs = failed['predictor_id'].to_list()
		failed_runs_unique = list(set(failed_runs))
		# Here the `~` inverts it to mean `.isNotIn()`
		df_passed = df[~df['predictor_id'].isin(failed_runs_unique)]
		df_passed = df_passed.round(3)
		dataframe = df_passed[['predictor_id', 'split', 'loss', name_metric_2]]

		if dataframe.empty:
			print("Yikes - There are no models that met the criteria specified.")
		else:
			Plot().performance(dataframe=dataframe)




class Jobset(BaseModel):
	"""
	- Used to group cross-fold Jobs.
	- Union of Hyperparamcombo, Foldset, and Queue.
	"""
	repeat_count = IntegerField()

	foldset = ForeignKeyField(Foldset, backref='jobsets')
	hyperparamcombo = ForeignKeyField(Hyperparamcombo, backref='jobsets')
	queue = ForeignKeyField(Queue, backref='jobsets')




class Job(BaseModel):
	"""
	- Gets its Algorithm through the Queue.
	- Saves its Model to a Predictor.
	"""
	repeat_count = IntegerField()
	#log = CharField() #catch & record stacktrace of failures and warnings?

	queue = ForeignKeyField(Queue, backref='jobs')
	hyperparamcombo = ForeignKeyField(Hyperparamcombo, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')
	fold = ForeignKeyField(Fold, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')
	jobset = ForeignKeyField(Jobset, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')


	def split_classification_metrics(labels_processed, predictions, probabilities, analysis_type):
		if (analysis_type == "classification_binary"):
			average = "binary"
			roc_average = "micro"
			roc_multi_class = None
		elif (analysis_type == "classification_multi"):
			average = "weighted"
			roc_average = "weighted"
			roc_multi_class = "ovr"
			
		split_metrics = {}
		# Let the classification_multi labels hit this metric in OHE format.
		split_metrics['roc_auc'] = sklearn.metrics.roc_auc_score(labels_processed, probabilities, average=roc_average, multi_class=roc_multi_class)
		# Then convert the classification_multi labels ordinal format.
		if (analysis_type == "classification_multi"):
			labels_processed = np.argmax(labels_processed, axis=1)

		split_metrics['accuracy'] = sklearn.metrics.accuracy_score(labels_processed, predictions)
		split_metrics['precision'] = sklearn.metrics.precision_score(labels_processed, predictions, average=average, zero_division=0)
		split_metrics['recall'] = sklearn.metrics.recall_score(labels_processed, predictions, average=average, zero_division=0)
		split_metrics['f1'] = sklearn.metrics.f1_score(labels_processed, predictions, average=average, zero_division=0)
		return split_metrics


	def split_regression_metrics(labels, predictions):
		split_metrics = {}
		split_metrics['r2'] = sklearn.metrics.r2_score(labels, predictions)
		split_metrics['mse'] = sklearn.metrics.mean_squared_error(labels, predictions)
		split_metrics['explained_variance'] = sklearn.metrics.explained_variance_score(labels, predictions)
		return split_metrics


	def split_classification_plots(labels_processed, predictions, probabilities, analysis_type):
		predictions = predictions.flatten()
		probabilities = probabilities.flatten()
		split_plot_data = {}
		
		if (analysis_type == "classification_binary"):
			labels_processed = labels_processed.flatten()
			split_plot_data['confusion_matrix'] = sklearn.metrics.confusion_matrix(labels_processed, predictions)
			fpr, tpr, _ = sklearn.metrics.roc_curve(labels_processed, probabilities)
			precision, recall, _ = sklearn.metrics.precision_recall_curve(labels_processed, probabilities)
		
		elif (analysis_type == "classification_multi"):
			# Flatten OHE labels for use with probabilities.
			labels_flat = labels_processed.flatten()
			fpr, tpr, _ = sklearn.metrics.roc_curve(labels_flat, probabilities)
			precision, recall, _ = sklearn.metrics.precision_recall_curve(labels_flat, probabilities)

			# Then convert unflat OHE to ordinal format for use with predictions.
			labels_ordinal = np.argmax(labels_processed, axis=1)
			split_plot_data['confusion_matrix'] = sklearn.metrics.confusion_matrix(labels_ordinal, predictions)

		split_plot_data['roc_curve'] = {}
		split_plot_data['roc_curve']['fpr'] = fpr
		split_plot_data['roc_curve']['tpr'] = tpr
		split_plot_data['precision_recall_curve'] = {}
		split_plot_data['precision_recall_curve']['precision'] = precision
		split_plot_data['precision_recall_curve']['recall'] = recall
		return split_plot_data


	def encoder_fit_labels(
		arr_labels:object, samples_train:list,
		labelcoder:object
	):
		"""
		- All Label columns are always used during encoding.
		- Rows determine what fit happens.
		"""
		if (labelcoder is not None):
			preproc = labelcoder.sklearn_preprocess

			if (labelcoder.only_fit_train == True):
				labels_to_fit = arr_labels[samples_train]
			elif (labelcoder.only_fit_train == False):
				labels_to_fit = arr_labels
				
			fitted_coders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
				sklearn_preprocess = preproc
				, samples_to_fit = labels_to_fit
			)
			# Save the fit.
			fitted_encoders = fitted_coders[0]#take out of list before adding to dict.
		return fitted_encoders


	def encoder_transform_labels(
		arr_labels:object,
		fitted_encoders:object, labelcoder:object 
	):
		encoding_dimension = labelcoder.encoding_dimension
		
		arr_labels = Labelcoder.transform_dynamicDimensions(
			fitted_encoders = [fitted_encoders] # `list(fitted_encoders)`, fails.
			, encoding_dimension = encoding_dimension
			, samples_to_transform = arr_labels
		)
		return arr_labels


	def colIndices_from_colNames(column_names:list, desired_cols:list):
		desired_cols = listify(desired_cols)
		col_indices = [column_names.index(c) for c in desired_cols]
		return col_indices

	def cols_by_indices(arr:object, col_indices:list):
		# Input and output 2D array. Fetches a subset of columns using their indices.
		# In the future if this needs to be adjusted to handle 3D array `[:,col_indices,:]`.
		subset_arr = arr[:,col_indices]
		return subset_arr


	def encoderset_fit_features(
		arr_features:object, samples_train:list,
		encoderset:object
	):
		featurecoders = list(encoderset.featurecoders)
		fitted_encoders = []
		if (len(featurecoders) > 0):
			f_cols = encoderset.feature.columns
			
			# For each featurecoder: fetch, transform, & concatenate matching features.
			# One nested list per Featurecoder. List of lists.
			for featurecoder in featurecoders:
				preproc = featurecoder.sklearn_preprocess

				if (featurecoder.only_fit_train == True):
					features_to_fit = arr_features[samples_train]
				elif (featurecoder.only_fit_train == False):
					features_to_fit = arr_features
				
				# Handles `Dataset.Sequence` by stacking the 2D arrays into a tall 2D array.
				features_shape = features_to_fit.shape
				if (len(features_shape)==3):
					rows_2D = features_shape[0] * features_shape[1]
					features_to_fit = features_to_fit.reshape(rows_2D, features_shape[2])

				# Only fit these columns.
				matching_columns = featurecoder.matching_columns
				# Get the indices of the desired columns.
				col_indices = Job.colIndices_from_colNames(
					column_names=f_cols, desired_cols=matching_columns
				)
				# Filter the array using those indices.
				features_to_fit = Job.cols_by_indices(features_to_fit, col_indices)

				# Fit the encoder on the subset.
				fitted_coders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
					sklearn_preprocess = preproc
					, samples_to_fit = features_to_fit
				)
				fitted_encoders.append(fitted_coders)
		return fitted_encoders


	def encoderset_transform_features(
		arr_features:object,
		fitted_encoders:list, encoderset:object 
	):
		"""
		- Can't overwrite columns with data of different type (e.g. encoding object to int), 
		  so they have to be pieced together.
		"""
		featurecoders = list(encoderset.featurecoders)
		if (len(featurecoders) > 0):
			# Handle Sequence (part 1): reshape 3D to tall 2D for transformation.
			features_shape = arr_features.shape
			if (len(features_shape)==3):
				rows_2D = features_shape[0] * features_shape[1]
				arr_features = arr_features.reshape(rows_2D, features_shape[2])

			f_cols = encoderset.feature.columns
			transformed_features = None #Used as a placeholder for `np.concatenate`.
			for featurecoder in featurecoders:
				idx = featurecoder.featurecoder_index
				fitted_coders = fitted_encoders[idx]# returns list
				encoding_dimension = featurecoder.encoding_dimension
				
				# Only transform these columns.
				matching_columns = featurecoder.matching_columns
				# Get the indices of the desired columns.
				col_indices = Job.colIndices_from_colNames(
					column_names=f_cols, desired_cols=matching_columns
				)
				# Filter the array using those indices.
				features_to_transform = Job.cols_by_indices(arr_features, col_indices)

				if (idx == 0):
					# It's the first encoder. Nothing to concat with, so just overwite the None value.
					transformed_features = Labelcoder.transform_dynamicDimensions(
						fitted_encoders = fitted_coders
						, encoding_dimension = encoding_dimension
						, samples_to_transform = features_to_transform
					)
				elif (idx > 0):
					encoded_features = Labelcoder.transform_dynamicDimensions(
						fitted_encoders = fitted_coders
						, encoding_dimension = encoding_dimension
						, samples_to_transform = features_to_transform
					)
					# Then concatenate w previously encoded features.
					transformed_features = np.concatenate(
						(transformed_features, encoded_features)
						, axis = 1
					)
			
			# After all featurecoders run, merge in leftover, unencoded columns.
			leftover_columns = featurecoders[-1].leftover_columns
			if (len(leftover_columns) > 0):
				# Get the indices of the desired columns.
				col_indices = Job.colIndices_from_colNames(
					column_names=f_cols, desired_cols=leftover_columns
				)
				# Filter the array using those indices.
				leftover_features = Job.cols_by_indices(arr_features, col_indices)
						
				transformed_features = np.concatenate(
					(transformed_features, leftover_features)
					, axis = 1
				)
			# Handle Sequence (part 2): reshape 3D to tall 2D for transformation.
			if (len(features_shape)==3):
				transformed_features = arr_features.reshape(
					features_shape[0],
					features_shape[1],
					features_shape[2]
				)
				
		elif (len(featurecoders) == 0):
			transformed_features = arr_features
		
		return transformed_features


	def predict(samples:dict, predictor_id:int, splitset_id:int=None):
		"""
		Evaluation: predictions, metrics, charts for each split/fold.
		- Metrics are run against encoded data because they won't accept string data.
		- `splitset_id` refers to a splitset provided for inference, not training.
		"""
		predictor = Predictor.get_by_id(predictor_id)
		hyperparamcombo = predictor.job.hyperparamcombo
		algorithm = predictor.job.queue.algorithm
		library = algorithm.library
		analysis_type = algorithm.analysis_type

		# Access the 2nd level of the `samples:dict` to determine if it has Labels.
		first_key = list(samples.keys())[0]
		if ('labels' in samples[first_key].keys()):
			has_labels = True
		else:
			has_labels = False

		# Prepare the logic.
		model = predictor.get_model()
		if (algorithm.library == 'keras'):
			model = predictor.get_model()
		elif (algorithm.library == 'pytorch'):
			# Returns tuple(model,optimizer)
			model = predictor.get_model()
			model = model[0].eval()
		fn_predict = dill_deserialize(algorithm.fn_predict)
		
		if (hyperparamcombo is not None):
			hp = hyperparamcombo.hyperparameters
		elif (hyperparamcombo is None):
			hp = {} #`**` cannot be None.

		if (has_labels == True):
			fn_lose = dill_deserialize(algorithm.fn_lose)
			loser = fn_lose(**hp)
			if (loser is None):
				raise ValueError("\nYikes - `fn_lose` returned `None`.\nDid you include `return loser` at the end of the function?\n")

		predictions = {}
		probabilities = {}
		if (has_labels == True):
			metrics = {}
			plot_data = {}

		if ("classification" in analysis_type):
			for split, data in samples.items():
				preds, probs = fn_predict(model, data)
				predictions[split] = preds
				probabilities[split] = probs
				# Outputs numpy.

				if (has_labels == True):
					# https://keras.io/api/losses/probabilistic_losses/
					if (library == 'keras'):
						loss = loser(data['labels'], probs)
					elif (library == 'pytorch'):
						tz_probs = torch.FloatTensor(probs)
						if (algorithm.analysis_type == 'classification_binary'):
							loss = loser(tz_probs, data['labels'])
							# convert back to numpy for metrics and plots.
							data['labels'] = data['labels'].detach().numpy()
						elif (algorithm.analysis_type == 'classification_multi'):
							flat_labels = data['labels'].flatten().to(torch.long)
							loss = loser(tz_probs, flat_labels)
							# convert back to *OHE* numpy for metrics and plots.
							data['labels'] = data['labels'].detach().numpy()
							data['labels'] = keras.utils.to_categorical(data['labels'])

					metrics[split] = Job.split_classification_metrics(
						data['labels'], preds, probs, analysis_type
					)
					metrics[split]['loss'] = float(loss)

					plot_data[split] = Job.split_classification_plots(
						data['labels'], preds, probs, analysis_type
					)
				
				# During prediction Keras OHE output gets made ordinal for metrics.
				# Use the probabilities to recreate the OHE so they can be inverse_transform'ed.
				if (("multi" in analysis_type) and (library == 'keras')):
					predictions[split] = []
					for p in probs:
						marker_position = np.argmax(p, axis=-1)
						empty_arr = np.zeros(len(p))
						empty_arr[marker_position] = 1
						predictions[split].append(empty_arr)
					predictions[split] = np.array(predictions[split])

		elif (analysis_type == "regression"):
			# The raw output values *is* the continuous prediction itself.
			probs = None
			for split, data in samples.items():
				preds = fn_predict(model, data)
				predictions[split] = preds
				# Outputs numpy.

				#https://keras.io/api/losses/regression_losses/
				if (has_labels == True):
					if (library == 'keras'):
						loss = loser(data['labels'], preds)
					elif (library == 'pytorch'):
						tz_preds = torch.FloatTensor(preds)
						loss = loser(tz_preds, data['labels'])
						# After obtaining loss, make labels numpy again for metrics.
						data['labels'] = data['labels'].detach().numpy()
						# `preds` object is still numpy.

					# Numpy inputs.
					metrics[split] = Job.split_regression_metrics(
						data['labels'], preds
					)
					metrics[split]['loss'] = float(loss)
				plot_data = None

		"""
		4b. Format predictions for saving.
		- Decode predictions before saving.
		- Doesn't use any Label data, but does use Labelcoder fit on the original Labels.
		"""
		labelcoder, fitted_encoders = Predictor.get_fitted_labelcoder(
			job=predictor.job, label=predictor.job.queue.splitset.label
		)

		if ((fitted_encoders is not None) and (hasattr(fitted_encoders, 'inverse_transform'))):
			for split, data in predictions.items():
				# OHE is arriving here as ordinal, not OHE.
				data = Labelcoder.if_1d_make_2d(data)
				predictions[split] = fitted_encoders.inverse_transform(data)
		elif((fitted_encoders is not None) and (not hasattr(fitted_encoders, 'inverse_transform'))):
			print(dedent("""
				Warning - `Predictor.predictions` are encoded. 
				They cannot be decoded because the `sklearn.preprocessing`
				encoder used does not have `inverse_transform`.
			"""))
		# Flatten.
		for split, data in predictions.items():
			if (data.ndim > 1):
				predictions[split] = data.flatten()

		if (has_labels == True):
			# 4c. Aggregate metrics across splits/ folds.
			# Alphabetize metrics dictionary by key.
			for k,v in metrics.items():
				metrics[k] = dict(natsorted(v.items()))
			# Aggregate metrics across splits (e.g. mean, pstdev).
			metric_names = list(list(metrics.values())[0].keys())
			metrics_aggregate = {}
			for metric in metric_names:
				split_values = []
				for split, split_metrics in metrics.items():
					# ran into obscure errors with `pstdev` when not `float(value)`
					value = float(split_metrics[metric])
					split_values.append(value)

				mean = statistics.mean(split_values)
				median = statistics.median(split_values)
				pstdev = statistics.pstdev(split_values)
				minimum = min(split_values)
				maximum = max(split_values)

				metrics_aggregate[metric] = {
					"mean":mean, "median":median, "pstdev":pstdev, 
					"minimum":minimum, "maximum":maximum 
				}
		
		if ((probs is not None) and ("multi" not in algorithm.analysis_type)):
			# Don't flatten the softmax probabilities.
			probabilities[split] = probabilities[split].flatten()

		if (has_labels == False):
			metrics = None
			metrics_aggregate = None
			plot_data = None

		if (splitset_id is not None):
			splitset = Splitset.get_by_id(splitset_id)
		else:
			splitset = None

		prediction = Prediction.create(
			predictions = predictions
			, probabilities = probabilities
			, metrics = metrics
			, metrics_aggregate = metrics_aggregate
			, plot_data = plot_data
			, predictor = predictor
			, splitset = splitset
		)
		return prediction


	def run(id:int, repeat_index:int, verbose:bool=False):
		"""
		Needs optimization = https://github.com/aiqc/aiqc/projects/1
		"""
		time_started = datetime.datetime.now()
		job = Job.get_by_id(id)
		if verbose:
			print(f"\nJob #{job.id} starting...")
		queue = job.queue
		algorithm = queue.algorithm
		analysis_type = algorithm.analysis_type
		library = algorithm.library
		hide_test = queue.hide_test
		splitset = queue.splitset
		hyperparamcombo = job.hyperparamcombo
		fold = job.fold
		"""
		1. Determines which splits/folds are needed.
		- Source of the training & evaluation data varies based on how Splitset and Foldset were designed.
		- The rest of the tasks in Job.run() look to `samples:dict` for their data.
		- The `key_*` variables are passed to downstream tasks. `key_train` could be either
		  'train' or 'folds_train_combined'.
		"""
		samples = {}
		if (hide_test == False):
			samples['test'] = splitset.samples['test']
			key_evaluation = 'test'
		elif (hide_test == True):
			key_evaluation = None

		if (splitset.has_validation):
			samples['validation'] = splitset.samples['validation']
			key_evaluation = 'validation'

		if (fold is not None):
			foldset = fold.foldset
			fold_index = fold.fold_index
			fold_samples = foldset.folds[fold_index].samples
			samples['folds_train_combined'] = fold_samples['folds_train_combined']
			samples['fold_validation'] = fold_samples['fold_validation']

			key_train = "folds_train_combined"
			key_evaluation = "fold_validation"
		elif (fold is None):
			samples['train'] = splitset.samples['train']
			key_train = "train"
		"""
		2. Encodes the labels and features.
		- Remember, you `.fit()` on either training data or all data (categoricals).
		- Then you transform the entire dataset because downstream processes may need the entire dataset:
		  e.g. fit imputer to training data, but then impute entire dataset so that encoders can use entire dataset.
		- So we transform the entire dataset, then divide it into splits/ folds.
		- Then we convert the arrays to pytorch tensors if necessary. Subsetting with a list of indeces and `shape`
		  work the same in both numpy and torch.
		"""
		# Labels - fetch and encode.
		if (splitset.supervision == "supervised"):
			arr_labels = splitset.label.to_numpy()
			labelcoder = splitset.label.get_latest_labelcoder()
			if (labelcoder is not None):
				fitted_encoders = Job.encoder_fit_labels(
					arr_labels=arr_labels, samples_train=samples[key_train],
					labelcoder=labelcoder
				)
				
				arr_labels = Job.encoder_transform_labels(
					arr_labels=arr_labels,
					fitted_encoders=fitted_encoders, labelcoder=labelcoder
				)
				FittedLabelcoder.create(fitted_encoders=fitted_encoders, job=job, labelcoder=labelcoder)
			if (library == 'pytorch'):
				arr_labels = torch.FloatTensor(arr_labels)

		# Features - fetch and encode.
		featureset = splitset.get_features()
		feature_count = len(featureset)
		features = []# expecting diff array shapes inside so it has to be list, not array.
		
		for feature in featureset:
			arr_features = feature.to_numpy()
			encoderset = feature.get_latest_encoderset()

			if (encoderset is not None):
				# This takes the entire array because it handles all features and splits.
				fitted_encoders = Job.encoderset_fit_features(
					arr_features=arr_features, samples_train=samples[key_train],
					encoderset=encoderset
				)

				arr_features = Job.encoderset_transform_features(
					arr_features=arr_features,
					fitted_encoders=fitted_encoders, encoderset=encoderset
				)
				FittedEncoderset.create(fitted_encoders=fitted_encoders, job=job, encoderset=encoderset)
			if (library == 'pytorch'):
				arr_features = torch.FloatTensor(arr_features)
			# Don't use the list if you don't have to.
			if (feature_count > 1):
				features.append(arr_features)			
		"""
		- Stage preprocessed data to be passed into the remaining Job steps.
		- Example samples dict entry: samples['train']['labels']
		- For each entry in the dict, fetch the rows from the encoded data.
		- Keras multi-input models accept input as a list. Not using nested dict for multiple
		  features because it would be hard to figure out feature.id-based keys on the fly.
		""" 
		for split, rows in samples.items():
			if (feature_count == 1):
				samples[split] = {
					"features": arr_features[rows]
					, "labels": arr_labels[rows]
				}
			elif (feature_count > 1):
				samples[split] = {
					"features": [arr_features[rows] for arr_features in features]
					, "labels": arr_labels[rows]
				}
		"""
		- Input shapes can only be determined after encoding has taken place.
		- `[0]` accessess the first sample in each array.
		- Does not impact the training loop's `batch_size`.
		- Shapes are used later by `get_model()` to initialize it.
		"""
		label_shape = samples[key_train]['labels'][0].shape
		if (feature_count == 1):
			features_shape = samples[key_train]['features'][0].shape
		elif (feature_count > 1):
			features_shape = [arr_features[0].shape for arr_features in samples[key_train]['features']]

		input_shapes = {
			"features_shape": features_shape
			, "label_shape": label_shape
		}
		"""
		3. Build and Train model.
		- This does not need to be modularized out of `Job.run()` because models are not
		  trained anywhere else in the codebase.
		"""
		if (hyperparamcombo is not None):
			hp = hyperparamcombo.hyperparameters
		elif (hyperparamcombo is None):
			hp = {} #`**` cannot be None.

		fn_build = dill_deserialize(algorithm.fn_build)
		if (splitset.supervision == "supervised"):
			# pytorch multiclass has a single ordinal label.
			if ((analysis_type == 'classification_multi') and (library == 'pytorch')):
				num_classes = len(splitset.label.unique_classes)
				model = fn_build(features_shape, num_classes, **hp)
			else:
				model = fn_build(features_shape, label_shape, **hp)
		elif (splitset.supervision == "unsupervised"):
			model = fn_build(features_shape, **hp)
		if (model is None):
			raise ValueError("\nYikes - `fn_build` returned `None`.\nDid you include `return model` at the end of the function?\n")
		
		# The model and optimizer get combined during training.
		fn_lose = dill_deserialize(algorithm.fn_lose)
		fn_optimize = dill_deserialize(algorithm.fn_optimize)
		fn_train = dill_deserialize(algorithm.fn_train)

		loser = fn_lose(**hp)
		if (loser is None):
			raise ValueError("\nYikes - `fn_lose` returned `None`.\nDid you include `return loser` at the end of the function?\n")

		if (library == 'keras'):
			optimizer = fn_optimize(**hp)
		elif (library == 'pytorch'):
			optimizer = fn_optimize(model, **hp)
		if (optimizer is None):
			raise ValueError("\nYikes - `fn_optimize` returned `None`.\nDid you include `return optimizer` at the end of the function?\n")

		
		if (key_evaluation is not None):
			samples_eval = samples[key_evaluation]
		elif (key_evaluation is None):
			samples_eval = None
		
		if (library == "keras"):
			model = fn_train(
				model = model
				, loser = loser
				, optimizer = optimizer
				, samples_train = samples[key_train]
				, samples_evaluate = samples_eval
				, **hp
			)
			if (model is None):
				raise ValueError("\nYikes - `fn_train` returned `model==None`.\nDid you include `return model` at the end of the function?\n")

			# Save the artifacts of the trained model.
			# If blank this value is `{}` not None.
			history = model.history.history
			"""
			- As of: Python(3.8.7), h5py(2.10.0), Keras(2.4.3), tensorflow(2.4.1)
			  model.save(buffer) working for neither `io.BytesIO()` nor `tempfile.TemporaryFile()`
			  https://github.com/keras-team/keras/issues/14411
			- So let's switch to a real file in appdirs.
			- Assuming `model.save()` will trigger OS-specific h5 drivers.
			"""
			# Write it.
			temp_file_name = f"{app_dir}temp_keras_model.h5"
			model.save(
				temp_file_name
				, include_optimizer = True
				, save_format = 'h5'
			)
			# Fetch the bytes ('rb': read binary)
			with open(temp_file_name, 'rb') as file:
				model_blob = file.read()
			os.remove(temp_file_name)

		elif (library == "pytorch"):
			model, history = fn_train(
				model = model
				, loser = loser
				, optimizer = optimizer
				, samples_train = samples[key_train]
				, samples_evaluate = samples_eval
				, **hp
			)
			if (model is None):
				raise ValueError("\nYikes - `fn_train` returned `model==None`.\nDid you include `return model` at the end of the function?\n")
			if (history is None):
				raise ValueError("\nYikes - `fn_train` returned `history==None`.\nDid you include `return model, history` the end of the function?\n")
			# Save the artifacts of the trained model.
			# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
			model_blob = io.BytesIO()
			torch.save(
				{
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()
				},
				model_blob
			)
			model_blob = model_blob.getvalue()
		"""
		5. Save everything to Predictor object.
		"""
		time_succeeded = datetime.datetime.now()
		time_duration = (time_succeeded - time_started).seconds

		# There's a chance that a duplicate job-repeat_index pair was running elsewhere and finished first.
		matching_predictor = Predictor.select().join(Job).join(Queue).where(
			Queue.id==queue.id, Job.id==job.id, Predictor.repeat_index==repeat_index)
		if (len(matching_predictor) > 0):
			raise ValueError(f"""
				Yikes - Duplicate run detected:
				Queue<{queue.id}>, Job<{job.id}>, Job.repeat_index<{repeat_index}>.
				Cancelling this instance of `run_jobs()` as there is another `run_jobs()` ongoing.
				No action needed, the other instance will continue running to completion.
			""")

		predictor = Predictor.create(
			time_started = time_started
			, time_succeeded = time_succeeded
			, time_duration = time_duration
			, model_file = model_blob
			, input_shapes = input_shapes
			, history = history
			, job = job
			, repeat_index = repeat_index
		)
		
		try:
			Job.predict(samples=samples, predictor_id=predictor.id)
		except:
			predictor.delete_instance()
			raise
		
		# Just to be sure not held in memory or multiprocess forked on a 2nd Queue.
		del samples
		del model
		return job


def execute_jobs(job_statuses:list, verbose:bool=False):  
	"""
	- This needs to be a top level function, otherwise you get pickle attribute error.
	- Alternatively, you can put this is a separate submodule file, and call it via
	  `import aiqc.execute_jobs.execute_jobs`
	- Tried `mp.Manager` and `mp.Value` for shared variable for progress, but gave up after
	  a full day of troubleshooting.
	- Also you have to get a separate database connection for the separate process.
	"""
	BaseModel._meta.database.close()
	BaseModel._meta.database = get_db()
	for j in tqdm(
		job_statuses
		, desc = "🔮 Training Models 🔮"
		, ncols = 100
	):
		if (j['predictor_id'] is None):
			Job.run(id=j['job_id'], verbose=verbose, repeat_index=j['repeat_index'])




class FittedEncoderset(BaseModel):
	"""
	- Job uses this to save the fitted_encoders, which are later used for inference.
	- Useful for accessing featurecoders for matching_columns, dimensions.
	- When I added support for multiple Features, updating `Job.fitted_encoders` during
	  `Job.run()` started to get unmanageable. Especially when you consider that not every
	  Feature type is guaranteed to have an Encoderset.
	"""
	fitted_encoders = PickleField()

	job = ForeignKeyField(Job, backref='fittedencodersets')
	encoderset = ForeignKeyField(Encoderset, backref='fittedencodersets')




class FittedLabelcoder(BaseModel):
	"""
	- See notes about FittedEncoderset.
	"""
	fitted_encoders = PickleField()

	job = ForeignKeyField(Job, backref='fittedlabelcoders')
	labelcoder = ForeignKeyField(Labelcoder, backref='fittedlabelcoders')



class Predictor(BaseModel):
	"""
	- This was refactored from "Predictor" to "Predictor"
	- Regarding metrics, the label encoder was fit on training split labels.
	"""
	repeat_index = IntegerField()
	time_started = DateTimeField()
	time_succeeded = DateTimeField()
	time_duration = IntegerField()
	model_file = BlobField()
	input_shapes = JSONField() # used by get_model()
	history = JSONField()

	job = ForeignKeyField(Job, backref='predictors')


	def get_model(id:int):
		predictor = Predictor.get_by_id(id)
		algorithm = predictor.job.queue.algorithm
		model_blob = predictor.model_file

		if (algorithm.library == "keras"):
			#https://www.tensorflow.org/guide/keras/save_and_serialize
			temp_file_name = f"{app_dir}temp_keras_model.h5"
			# Workaround: write bytes to file so keras can read from path instead of buffer.
			with open(temp_file_name, 'wb') as f:
				f.write(model_blob)
			h5 = h5py.File(temp_file_name, 'r')
			model = keras.models.load_model(h5, compile=True)
			os.remove(temp_file_name)
			# Unlike pytorch, it's doesn't look like you need to initialize the optimizer or anything.
			return model

		elif (algorithm.library == 'pytorch'):
			# https://pytorch.org/tutorials/beginner/saving_loading_models.html#load
			# Need to initialize the classes first, which requires reconstructing them.
			if (predictor.job.hyperparamcombo is not None):
				hp = predictor.job.hyperparamcombo.hyperparameters
			elif (predictor.job.hyperparamcombo is None):
				hp = {}
			features_shape = predictor.input_shapes['features_shape']
			label_shape = predictor.input_shapes['label_shape']

			fn_build = dill_deserialize(algorithm.fn_build)
			fn_optimize = dill_deserialize(algorithm.fn_optimize)

			if (algorithm.analysis_type == 'classification_multi'):
				num_classes = len(predictor.job.queue.splitset.label.unique_classes)
				model = fn_build(features_shape, num_classes, **hp)
			else:
				model = fn_build(features_shape, label_shape, **hp)
			
			optimizer = fn_optimize(model, **hp)

			model_bytes = io.BytesIO(model_blob)
			checkpoint = torch.load(model_bytes)
			# Don't assign them: `model = model.load_state_dict ...`
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			# "must call model.eval() to set dropout & batchNorm layers to evaluation mode before prediction." 
			# ^ but you don't need to pass any data into eval()
			return model, optimizer

	def export_model(id:int, file_path:str=None):
		predictor = Predictor.get_by_id(id)
		algorithm = predictor.job.queue.algorithm
		
		if (file_path is None):
			dtime = datetime.datetime.now().strftime('%Y%b%d_%H:%M')
			if (algorithm.library == "keras"):
				ext = '.h5'
			elif (algorithm.library == 'pytorch'):
				ext = '.pt'
			file_path = f"{app_dir}/models/predictor{predictor.id}_model({dtime}){ext}"
		
		file_path = os.path.abspath(file_path)
		folder = f"{app_dir}/models"
		os.makedirs(folder, exist_ok=True)

		# We already have the bytes of the file we need to write.
		model_blob = predictor.model_file
		# trying `+` because directory may not exist yet.
		with open(file_path, 'wb+') as f:
			f.write(model_blob)
			f.close()

		os.path.exists(file_path)
		print(dedent(
			f"\nModel exported to the following absolute path:" \
			f"\n{file_path}\n"
		))
		return file_path


	def get_hyperparameters(id:int, as_pandas:bool=False):
		"""This is actually a method of `Hyperparamcombo` so we just pass through."""
		predictor = Predictor.get_by_id(id)
		hyperparamcombo = predictor.job.hyperparamcombo
		hp = hyperparamcombo.get_hyperparameters(as_pandas=as_pandas)
		return hp

		
	def plot_learning_curve(id:int, loss_skip_15pct:bool=False):
		predictor = Predictor.get_by_id(id)
		algorithm = predictor.job.queue.algorithm
		analysis_type = algorithm.analysis_type

		history = predictor.history
		dataframe = pd.DataFrame.from_dict(history, orient='index').transpose()
		Plot().learning_curve(
			dataframe = dataframe
			, analysis_type = analysis_type
			, loss_skip_15pct = loss_skip_15pct
		)


	def tabular_schemas_match(set_original, set_new):
		# Set can be either Label or Feature. Needs `columns` and `.get_dtypes`.
		cols_old = set_original.columns
		cols_new = set_new.columns
		if (cols_new != cols_old):
			raise ValueError("\nYikes - New columns do not match original columns.\n")

		typs_old = set_original.get_dtypes()
		typs_new = set_new.get_dtypes()
		if (typs_new != typs_old):
			raise ValueError(dedent("""
				Yikes - New dtypes do not match original dtypes.
				The Low-Level API methods for Dataset creation accept a `dtype` argument to fix this.
			"""))

	def image_schemas_match(feature_old, feature_new):
		image_old = feature_old.dataset.files[0].images[0]
		image_new = feature_new.dataset.files[0].images[0]
		if (image_old.size != image_new.size):
			raise ValueError(f"\nYikes - The new image size:{image_new.size} did not match the original image size:{image_old.size}.\n")
		if (image_old.mode != image_new.mode):
			raise ValueError(f"\nYikes - The new image color mode:{image_new.mode} did not match the original image color mode:{image_old.mode}.\n")
			

	def schemaNew_matches_schemaOld(splitset_new:object, splitset_old:object):
		# Get the new and old featuresets. Loop over them by index.
		features_new = splitset_new.get_features()
		features_old = splitset_old.get_features()

		if (len(features_new) != len(features_old)):
			raise ValueError("\nYikes - Your new and old Splitsets do not contain the same number of Features.\n")

		for i, feature_new in enumerate(features_new):
			feature_old = features_old[i]
			feature_old_typ = feature_old.dataset.dataset_type
			feature_new_typ = feature_new.dataset.dataset_type
			if (feature_old_typ != feature_new_typ):
				raise ValueError(f"\nYikes - New Feature dataset_type={feature_new_typ} != old Feature dataset_type={feature_old_typ}.\n")
			if ((feature_new_typ == 'tabular') or (feature_new_typ == 'sequence')):
				Predictor.tabular_schemas_match(feature_old, feature_new)
			elif (feature_new_typ == 'image'):
				Predictor.image_schemas_match(feature_old, feature_new)

		# Only verify Labels if the inference new Splitset provides Labels.
		# Otherwise, it may be conducting pure inference.
		label = splitset_new.label
		if (label is not None):
			label_new = label
			label_new_typ = label_new.dataset.dataset_type

			if (splitset_old.supervision == 'unsupervised'):
				raise ValueError("\nYikes - New Splitset has Labels, but old Splitset does not have Labels.\n")

			elif (splitset_old.supervision == 'supervised'):
				label_old =  splitset_old.label
				label_old_typ = label_old.dataset.dataset_type
			
			if (label_old_typ != label_new_typ):
				raise ValueError("\nYikes - New Label and original Label come from different `dataset_types`.\n")
			if (label_new_typ == 'tabular'):
				Predictor.tabular_schemas_match(label_old, label_new)


	def get_fitted_encoderset(job:object, feature:object):
		"""
		- Given a Feature, you want to know if it needs to be transformed,
		  and, if so, how to transform it.
		"""
		fitted_encodersets = FittedEncoderset.select().join(Encoderset).where(
			FittedEncoderset.job==job, FittedEncoderset.encoderset.feature==feature
		)

		if (not fitted_encodersets):
			return None, None
		else:
			encoderset = fitted_encodersets[0].encoderset
			fitted_encoders = fitted_encodersets[0].fitted_encoders
			return encoderset, fitted_encoders


	def get_fitted_labelcoder(job:object, label:object):
		"""
		- Given a Feature, you want to know if it needs to be transformed,
		  and, if so, how to transform it.
		"""
		fitted_labelcoders = FittedLabelcoder.select().join(Labelcoder).where(
			FittedLabelcoder.job==job, FittedLabelcoder.labelcoder.label==label
		)
		if (not fitted_labelcoders):
			return None, None
		else:
			labelcoder = fitted_labelcoders[0].labelcoder
			fitted_encoders = fitted_labelcoders[0].fitted_encoders
			return labelcoder, fitted_encoders

			
	def infer(id:int, splitset_id:int):
		"""
		- Splitset is used because Labels and Features can come from different types of Datasets.
		- Verifies both Features and Labels match original schema.
		"""
		splitset_new = Splitset.get_by_id(splitset_id)
		predictor = Predictor.get_by_id(id)
		splitset_old = predictor.job.queue.splitset

		Predictor.schemaNew_matches_schemaOld(splitset_new, splitset_old)
		library = predictor.job.queue.algorithm.library

		featureset_new = splitset_new.get_features()
		featureset_old = splitset_old.get_features()
		feature_count = len(featureset_new)
		features = []# expecting different array shapes so it has to be list, not array.
		for i, feature_new in enumerate(featureset_new):
			arr_features = feature_new.to_numpy()
			encoderset, fitted_encoders = Predictor.get_fitted_encoderset(
				job=predictor.job, feature=featureset_old[i]
			)
			if (encoderset is not None):
				# Don't need to check types because Encoderset creation protects
				# against unencodable types.
				arr_features = Job.encoderset_transform_features(
					arr_features=arr_features,
					fitted_encoders=fitted_encoders, encoderset=encoderset
				)
			if (library == 'pytorch'):
				arr_features = torch.FloatTensor(arr_features)
			if (feature_count > 1):
				features.append(arr_features)
			else:
				# We don't need to do any row filtering so it can just be overwritten.
				features = arr_features
		"""
		- Pack into samples for the Algorithm functions.
		- This is two levels deep to mirror how the training samples were structured 
		  e.g. `samples[<trn,val,tst>]`
		- str() id because int keys aren't JSON serializable.
		"""
		str_id = str(splitset_id)
		samples = {str_id: {'features':features}}

		if (splitset_new.label is not None):
			label_new = splitset_new.label
			label_old = splitset_old.label
		else:
			label_new = None
			label_old = None

		if (label_new is not None):			
			arr_labels = label_new.to_numpy()	

			labelcoder, fitted_encoders = Predictor.get_fitted_labelcoder(
				job=predictor.job, label=label_old
			)
			if (labelcoder is not None):
				arr_labels = Job.encoder_transform_labels(
					arr_labels=arr_labels,
					fitted_encoders=fitted_encoders, labelcoder=labelcoder
				)
			if (library == 'pytorch'):
				arr_labels = torch.FloatTensor(arr_labels)
			samples[str_id]['labels'] = arr_labels

		prediction = Job.predict(
			samples=samples, predictor_id=id, splitset_id=splitset_id
		)
		return prediction




class Prediction(BaseModel):
	"""
	- Many-to-Many for making predictions after of the training experiment.
	- We use the low level API to create a Dataset because there's a lot of formatting 
	  that happens during Dataset creation that we would lose out on with raw numpy/pandas 
	  input: e.g. columns may need autocreation, and who knows what connectors we'll have 
	  in the future. This forces us to  validate dtypes and columns after the fact.
	"""
	predictions = PickleField()
	probabilities = PickleField(null=True) # Not used for regression.
	metrics = PickleField(null=True) #inference
	metrics_aggregate = PickleField(null=True) #inference.
	plot_data = PickleField(null=True) # No regression-specific plots yet.

	predictor = ForeignKeyField(Predictor, backref='predictions')
	# dataset present if created for inference, v.s. null if from Original training set.
	splitset = ForeignKeyField(Splitset, deferrable='INITIALLY DEFERRED', null=True, backref='dataset') 

	"""
	- I moved these plots out of Predictor into Prediction because it felt weird to access the
	  Prediction via `predictions[0]`.
	- If we ever do non-deterministic algorithms then we would not have a 1-1 mapping 
	  between Predictor and Prediction.
	"""
	def plot_confusion_matrix(id:int):
		prediction = Prediction.get_by_id(id)
		prediction_plot_data = prediction.plot_data
		analysis_type = prediction.predictor.job.queue.algorithm.analysis_type
		if (analysis_type == "regression"):
			raise ValueError("\nYikes - <Algorithm.analysis_type> of 'regression' does not support this chart.\n")
		cm_by_split = {}

		labelcoder, fitted_encoders = Predictor.get_fitted_labelcoder(
			job=prediction.predictor.job, label=prediction.predictor.job.queue.splitset.label
		)
		if (labelcoder is not None):
			if hasattr(fitted_encoders,'categories_'):
				labels = list(fitted_encoders.categories_[0])
			elif hasattr(fitted_encoders,'classes_'):
				labels = fitted_encoders.classes_.tolist()
		else:
			unique_classes = prediction.predictor.job.queue.splitset.label.unique_classes
			labels = list(unique_classes)

		for split, data in prediction_plot_data.items():
			cm_by_split[split] = data['confusion_matrix']

		Plot().confusion_matrix(cm_by_split=cm_by_split, labels= labels)


	def plot_precision_recall(id:int):
		prediction = Prediction.get_by_id(id)
		predictor_plot_data = prediction.plot_data
		analysis_type = prediction.predictor.job.queue.algorithm.analysis_type
		if (analysis_type == "regression"):
			raise ValueError("\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n")

		pr_by_split = {}
		for split, data in predictor_plot_data.items():
			pr_by_split[split] = data['precision_recall_curve']

		dfs = []
		for split, data in pr_by_split.items():
			df = pd.DataFrame()
			df['precision'] = pd.Series(pr_by_split[split]['precision'])
			df['recall'] = pd.Series(pr_by_split[split]['recall'])
			df['split'] = split
			dfs.append(df)
		dataframe = pd.concat(dfs, ignore_index=True)
		dataframe = dataframe.round(3)

		Plot().precision_recall(dataframe=dataframe)


	def plot_roc_curve(id:int):
		prediction = Prediction.get_by_id(id)
		predictor_plot_data = prediction.plot_data
		analysis_type = prediction.predictor.job.queue.algorithm.analysis_type
		if (analysis_type == "regression"):
			raise ValueError("\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n")

		roc_by_split = {}
		for split, data in predictor_plot_data.items():
			roc_by_split[split] = data['roc_curve']

		dfs = []
		for split, data in roc_by_split.items():
			df = pd.DataFrame()
			df['fpr'] = pd.Series(roc_by_split[split]['fpr'])
			df['tpr'] = pd.Series(roc_by_split[split]['tpr'])
			df['split'] = split
			dfs.append(df)

		dataframe = pd.concat(dfs, ignore_index=True)
		dataframe = dataframe.round(3)

		Plot().roc_curve(dataframe=dataframe)


#==================================================
# MID-TRAINING CALLBACKS
#==================================================

class TrainingCallback():
	class Keras():
		class MetricCutoff(keras.callbacks.Callback):
			"""
			- Worried that these inner functions are not pickling during multi-processing.
			https://stackoverflow.com/a/8805244/5739514
			"""
			def __init__(self, thresholds:list):
				"""
				# Tested with keras:2.4.3, tensorflow:2.3.1
				# `thresholds` is list of dictionaries with 1 dict per metric.
				metrics_cuttoffs = [
					{"metric":"val_acc", "cutoff":0.94, "above_or_below":"above"},
					{"metric":"acc", "cutoff":0.90, "above_or_below":"above"},
					{"metric":"val_loss", "cutoff":0.26, "above_or_below":"below"},
					{"metric":"loss", "cutoff":0.30, "above_or_below":"below"},
				]
				# Only stops training early if all user-specified metrics are satisfied.
				# `above_or_below`: where 'above' means `>=` and 'below' means `<=`.
				"""
				self.thresholds = thresholds
				

			def on_epoch_end(self, epoch, logs=None):
				logs = logs or {}
				# Check each user-defined threshold to see if it is satisfied.
				for threshold in self.thresholds:
					metric = logs.get(threshold['metric'])
					if (metric is None):
						raise ValueError(dedent(f"""
						Yikes - The metric named '{threshold['metric']}' not found when running `logs.get('{threshold['metric']}')`
						during `TrainingCallback.Keras.MetricCutoff.on_epoch_end`.
						"""))
					cutoff = threshold['cutoff']

					above_or_below = threshold['above_or_below']
					if (above_or_below == 'above'):
						statement = operator.ge(metric, cutoff)
					elif (above_or_below == 'below'):
						statement = operator.le(metric, cutoff)
					else:
						raise ValueError(dedent(f"""
						Yikes - Value for key 'above_or_below' must be either string 'above' or 'below'.
						You provided:{above_or_below}
						"""))

					if (statement == False):
						break # Out of for loop.
						
				if (statement == False):
					pass # Thresholds not satisfied, so move on to the next epoch.
				elif (statement == True):
					# However, if the for loop actually finishes, then all metrics are satisfied.
					print(
						f":: Epoch #{epoch} ::\n" \
						f"Congratulations - satisfied early stopping thresholds defined in `MetricCutoff` callback:\n"\
						f"{pprint.pformat(self.thresholds)}\n"
					)
					self.model.stop_training = True


#==================================================
# HIGH LEVEL API 
#==================================================

class Pipeline():
	"""Create Dataset, Feature, Label, Splitset, and Foldset."""
	def parse_tabular_input(dataFrame_or_filePath:object, dtype:object=None):
		"""Create the dataset from either df or file."""
		d = dataFrame_or_filePath
		data_type = str(type(d))
		if (data_type == "<class 'pandas.core.frame.DataFrame'>"):
			dataset = Dataset.Tabular.from_pandas(dataframe=d, dtype=dtype)
		elif (data_type == "<class 'str'>"):
			if '.csv' in d:
				source_file_format='csv'
			elif '.tsv' in d:
				source_file_format='tsv'
			elif '.parquet' in d:
				source_file_format='parquet'
			else:
				raise ValueError(dedent("""
				Yikes - None of the following file extensions were found in the path you provided:
				'.csv', '.tsv', '.parquet'
				"""))
			dataset = Dataset.Tabular.from_path(
				file_path = d
				, source_file_format = source_file_format
				, dtype = dtype
			)
		else:
			raise ValueError("\nYikes - The `dataFrame_or_filePath` is neither a string nor a Pandas dataframe.\n")
		return dataset


	class Tabular():
		def make(
			dataFrame_or_filePath:object
			, dtype:object = None
			, label_column:str = None
			, features_excluded:list = None
			, label_encoder:object = None
			, feature_encoders:list = None
			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			features_excluded = listify(features_excluded)
			feature_encoders = listify(feature_encoders)

			dataset = Pipeline.parse_tabular_input(
				dataFrame_or_filePath = dataFrame_or_filePath
				, dtype = dtype
			)
			if (label_column is not None):
				label = dataset.make_label(columns=[label_column])
				label_id = label.id

				if (label_encoder is not None): 
					label.make_labelcoder(sklearn_preprocess=label_encoder)
			elif (label_column is None):
				# Needs to know if label exists so that it can exlcude it.
				label_id = None

			if (features_excluded is None):
				if (label_column is not None):
					feature = dataset.make_feature(exclude_columns=[label_column])
			elif (features_excluded is not None):
				feature = dataset.make_feature(exclude_columns=features_excluded)
			
			if (feature_encoders is not None):					
				encoderset = feature.make_encoderset()
				for fc in feature_encoders:
					encoderset.make_featurecoder(**fc)

			splitset = Splitset.make(
				feature_ids = [feature.id]
				, label_id = label_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (fold_count is not None):
				splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)

			return splitset


	class Sequence():
		def make(
			seq_ndarray3D:object
			, seq_dtype:object = None
			, seq_features_excluded:list = None
			, seq_feature_encoders:list = None
			
			, tab_DF_or_path:object = None
			, tab_dtype:object = None
			, tab_label_column:str = None
			, tab_label_encoder:object = None
			
			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			seq_features_excluded = listify(seq_features_excluded)
			seq_feature_encoders = listify(seq_feature_encoders)

			# ------ SEQUENCE FEATURE ------
			seq_dataset = Dataset.Sequence.from_numpy(
				ndarray_3D=seq_ndarray3D,
				dtype=seq_dtype
			)

			if (seq_features_excluded is not None):
				feature = seq_dataset.make_feature(exclude_columns=seq_features_excluded)
			elif (seq_features_excluded is None):
				feature = seq_dataset.make_feature()
			
			if (seq_feature_encoders is not None):					
				encoderset = feature.make_encoderset()
				for fc in seq_feature_encoders:
					encoderset.make_featurecoder(**fc)

			# ------ TABULAR LABEL ------
			if (
				((tab_DF_or_path is None) and (tab_label_column is not None))
				or
				((tab_DF_or_path is not None) and (tab_label_column is None))
			):
				raise ValueError("\nYikes - `tabularDF_or_path` and `label_column` are either used together or not at all.\n")

			if (tab_DF_or_path is not None):
				dataset_tabular = Pipeline.parse_tabular_input(
					dataFrame_or_filePath = tab_DF_or_path
					, dtype = tab_dtype
				)
				# Tabular-based Label.
				label = dataset_tabular.make_label(columns=[tab_label_column])
				label_id = label.id

				if (tab_label_encoder is not None): 
					label.make_labelcoder(sklearn_preprocess=tab_label_encoder)
			elif (tab_DF_or_path is None):
				label_id = None

			splitset = Splitset.make(
				feature_ids = [feature.id]
				, label_id = label_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (fold_count is not None):
				splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)

			return splitset


	class Image():
		def make(
			pillow_save:dict = {}
			, folderPath_or_urls:str = None
			, tabularDF_or_path:object = None
			, tabular_dtype:object = None
			, label_column:str = None
			, label_encoder:object = None
			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			if (isinstance(folderPath_or_urls, str)):
				dataset_image = Dataset.Image.from_folder(
					folder_path = folderPath_or_urls
					, pillow_save = pillow_save
				)
			elif (isinstance(folderPath_or_urls, list)):
				dataset_image = Dataset.Image.from_urls(
					urls = folderPath_or_urls
					, pillow_save = pillow_save
				)
			# Image-based Feature.
			feature = dataset_image.make_feature()

			if (
				((tabularDF_or_path is None) and (label_column is not None))
				or
				((tabularDF_or_path is not None) and (label_column is None))
			):
				raise ValueError("\nYikes - `tabularDF_or_path` and `label_column` are either used together or not at all.\n")

			# Dataset.Tabular
			if (tabularDF_or_path is not None):
				dataset_tabular = Pipeline.parse_tabular_input(
					dataFrame_or_filePath = tabularDF_or_path
					, dtype = tabular_dtype
				)
				# Tabular-based Label.
				label = dataset_tabular.make_label(columns=[label_column])
				label_id = label.id

				if (label_encoder is not None): 
					label.make_labelcoder(sklearn_preprocess=label_encoder)
			elif (tabularDF_or_path is None):
				label_id = None
			
			splitset = Splitset.make(
				feature_ids = [feature.id]
				, label_id = label_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (fold_count is not None):
				splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)
			return splitset


class Experiment():
	"""
	- Create Algorithm, Hyperparamset, Preprocess, and Queue.
	- Put Preprocess here because it's weird to encode labels before you know what your final training layer looks like.
	  Also, it's optional, so you'd have to access it from splitset before passing it in.
	- The only pre-existing things that need to be passed in are `splitset_id` and the optional `foldset_id`.


	`encoder_feature`: List of dictionaries describing each encoder to run along with filters for different feature columns.
	`encoder_label`: Single instantiation of an sklearn encoder: e.g. `OneHotEncoder()` that gets applied to the full label array.
	"""
	def make(
		library:str
		, analysis_type:str
		, fn_build:object
		, fn_train:object
		, splitset_id:int
		, repeat_count:int = 1
		, hide_test:bool = False
		, fn_optimize:object = None
		, fn_predict:object = None
		, fn_lose:object = None
		, hyperparameters:dict = None
		, pick_count = None
		, pick_percent = None
		, foldset_id:int = None
	):

		algorithm = Algorithm.make(
			library = library
			, analysis_type = analysis_type
			, fn_build = fn_build
			, fn_train = fn_train
			, fn_optimize = fn_optimize
			, fn_predict = fn_predict
			, fn_lose = fn_lose
		)

		if (hyperparameters is not None):
			hyperparamset = algorithm.make_hyperparamset(
				hyperparameters = hyperparameters
				, pick_count = pick_count
				, pick_percent = pick_percent
			)
			hyperparamset_id = hyperparamset.id
		elif (hyperparameters is None):
			hyperparamset_id = None

		queue = algorithm.make_queue(
			splitset_id = splitset_id
			, repeat_count = repeat_count
			, hide_test = hide_test
			, hyperparamset_id = hyperparamset_id
			, foldset_id = foldset_id
		)
		return queue
