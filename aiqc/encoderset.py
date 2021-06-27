from peewee import CharField, ForeignKeyField, IntegerField
from .basemodel import BaseModel
from .encoderset import Encoderset
from .feature import Feature
from .featurecoder import Featurecoder
from .utility import listify

class Encoderset(BaseModel):
	"""
	- Preprocessing should not happen prior to Dataset ingestion because you need to do it after the split to avoid bias.
	  For example, encoder.fit() only on training split - then .transform() train, validation, and test. 
	- Don't restrict a preprocess to a specific Algorithm. Many algorithms are created as different hyperparameters are tried.
	  Also, Preprocess is somewhat predetermined by the dtypes present in the Label and Feature.
	- Although Encoderset seems uneccessary, you need something to sequentially group the Featurecoders onto.
	- In future, maybe Labelcoder gets split out from Encoderset and it becomes Featurecoderset.
	"""
	encoder_count = IntegerField()
	description = CharField(null=True)

	feature = ForeignKeyField(Feature, backref='encodersets')

	def from_feature(
		feature_id:int
		, encoder_count:int = 0
		, description:str = None
	):
		feature = Feature.get_by_id(feature_id)
		encoderset = Encoderset.create(
			encoder_count = encoder_count
			, description = description
			, feature = feature
		)
		return encoderset


	def make_featurecoder(
		id:int
		, sklearn_preprocess:object
		, include:bool = True
		, verbose:bool = True
		, dtypes:list = None
		, columns:list = None
	):
		dtypes = listify(dtypes)
		columns = listify(columns)
		fc = Featurecoder.from_encoderset(
			encoderset_id = id
			, sklearn_preprocess = sklearn_preprocess
			, include = include
			, dtypes = dtypes
			, columns = columns
			, verbose = verbose
		)
		return fc