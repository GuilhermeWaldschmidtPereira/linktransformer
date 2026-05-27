# __init__.py

__version__ = "0.1.17"
__MODEL_HUB_ORGANIZATION__ = 'sentence-transformers' #For compatibility with sentence-transformers
from .data import DATA_DIR_PATH
from .infer_scann import *
from .preprocess import *

# Dependências de treino são opcionais em ambientes de inferência (ex.: imagem ScaNN)
try:
	from .train_model import *
except Exception:
	pass

try:
	from .modified_sbert import *
except Exception:
	pass

try:
	from .train_clf_model import train_clf_model
except Exception:
	train_clf_model = None

from .modelling.LinkTransformer import LinkTransformer
from .modelling.LinkTransformerClassifier import LinkTransformerClassifier


