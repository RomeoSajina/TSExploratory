from sklearn.exceptions import DataConversionWarning
import tensorflow as tf
import warnings
from core import Plotly
from core import Config, Metadata
from core import DataFactory

from core import PersistentModelWrapper
from core import SeasonalPersistentModelWrapper

from core import ARIMAModelWrapper
from core import ARMAModelWrapper
from core import ARModelWrapper
from core import MAModelWrapper
from core import SARIMAXModelWrapper
from core import UnobservedComponentsModelWrapper

from core import ProphetModelWrapper

from core import BidirectionalLSTMModelWrapper
from core import CNNLSTMModelWrapper
from core import GRUModelWrapper
from core import LSTMModelWrapper
from core import TimeDistributedCNNLSTMModelWrapper
from core import CustomModelWrapper
from core import CNNModelWrapper
from core import MultichannelCNNModelWrapper
from core import MLPModelWrapper
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

config = DataFactory.load_ts()
config.cddm = None
config.apply_metadata(Metadata.version_1())
#config.apply_metadata(Metadata.version_2())
#config.apply_metadata(Metadata.version_3())
config.apply_metadata(Metadata.version_4())



# Base models
model = PersistentModelWrapper(config)
model = SeasonalPersistentModelWrapper(config)

# Stats models
model = ARModelWrapper(config)
model = MAModelWrapper(config)
model = ARMAModelWrapper(config)
model = ARIMAModelWrapper(config)
model = SARIMAXModelWrapper(config)
model = UnobservedComponentsModelWrapper(config)

model = ProphetModelWrapper(config)

# NN models
model = BidirectionalLSTMModelWrapper(config)
model = CNNLSTMModelWrapper(config)
model = GRUModelWrapper(config)
model = LSTMModelWrapper(config)
model = TimeDistributedCNNLSTMModelWrapper(config)
model = CNNModelWrapper(config)
model = MultichannelCNNModelWrapper(config)
model = MLPModelWrapper(config)


#model.train_sequentially = False
#model.save_load_model = False
#config.n_outputs = 60
#config.n_outputs = 30
#config.n_outputs = 7

model.fit()
#model.fit_model()

model.plot_train()
#model.save_train_figure()

model.plot_predict_multiple()
model.plot_predict()
model.plot_predict(730)
#model.save_prediction_figure()
Plotly.ZOOM=-100
Plotly.ZOOM=None




"""
Name: TSExploratory

BUGOVITO:
BidirectionalLSTMModelWrapper
GRUModelWrapper
LSTMModelWrapper
"""

