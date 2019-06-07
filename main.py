from sklearn.exceptions import DataConversionWarning
import tensorflow as tf
import warnings
import datetime
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
from core import MultiCNNModelWrapper
from core import MLPModelWrapper
from core import AutoencoderMLPModelWrapper
from core import AutoencoderRandomDropoutMLPModelWrapper
from core import RandomDropoutLSTMModelWrapper
from core import AutoencoderRandomDropoutBidirectionalLSTMModelWrapper
from core import AutoencoderRandomDropoutTimeDistributedCNNLSTMModelWrapper
from core import AutoencoderRandomDropoutCNNLSTMModelWrapper
from core import AutoencoderCNNLSTMTimeDistributedModelWrapper
from core import RandomDropoutGRUModelWrapper
from core import AutoencoderRandomDropoutCNNModelWrapper
from core import AutoencoderRandomDropoutMultiCNNModelWrapper
from core import AutoencoderRandomDropoutMLPLSTMModelWrapper
from core import AutoencoderRandomDropoutMLPGRUModelWrapper
from core import ResNetClassificationModelWrapper
from core import ResNetLSTMModelWrapper

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

config = DataFactory.load_ts()
config.cddm = None

#config.apply_metadata(Metadata.version_1())
#config.apply_metadata(Metadata.version_2())
#config.apply_metadata(Metadata.version_3())
#config.apply_metadata(Metadata.version_4())


DataFactory.plot_yearly()

target_dates = [datetime.datetime(2018, 6, 1),
                datetime.datetime(2018, 7, 1),
                datetime.datetime(2018, 8, 1),
                datetime.datetime(2018, 9, 1)]

DataFactory.plot_for_target_dates(target_dates=target_dates, x_is_distance=True)
DataFactory.plot_for_target_dates(target_dates=target_dates, x_is_distance=False)

# Cijeli 7 mj
target_dates = [datetime.datetime(2018, 7, 1) + datetime.timedelta(x) for x in range(0, 31)]
DataFactory.plot_for_target_dates(target_dates=target_dates, x_is_distance=False)

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
model = MultiCNNModelWrapper(config)
model = MLPModelWrapper(config)
model = AutoencoderMLPModelWrapper(config)
model = AutoencoderRandomDropoutMLPModelWrapper(config)
model = RandomDropoutLSTMModelWrapper(config)
model = AutoencoderRandomDropoutBidirectionalLSTMModelWrapper(config)
model = AutoencoderRandomDropoutTimeDistributedCNNLSTMModelWrapper(config)
model = AutoencoderRandomDropoutCNNLSTMModelWrapper(config)
model = AutoencoderCNNLSTMTimeDistributedModelWrapper(config)
model = RandomDropoutGRUModelWrapper(config)
model = AutoencoderRandomDropoutCNNModelWrapper(config)
model = AutoencoderRandomDropoutMultiCNNModelWrapper(config)
model = AutoencoderRandomDropoutMLPLSTMModelWrapper(config)
model = AutoencoderRandomDropoutMLPGRUModelWrapper(config)
model = ResNetClassificationModelWrapper(config)
model = ResNetLSTMModelWrapper(config)

model.train_sequentially = False

model.fit()

model.plot_train()
#model.save_train_figure()

model.plot_predict_multiple()
model.plot_predict()

Plotly.ZOOM = -305
model.plot_predict(425)
#model.save_prediction_figure()


for i in range(1, 13):
    config.apply_metadata(getattr(Metadata, "version_"+str(i))())
    model = AutoencoderCNNLSTMTimeDistributedModelWrapper(config)
    model.epochs = 1500
    model.fit()


ALL = [
    BidirectionalLSTMModelWrapper,
    CNNLSTMModelWrapper,
    GRUModelWrapper,
    LSTMModelWrapper,
    TimeDistributedCNNLSTMModelWrapper,
    CNNModelWrapper,
    MultiCNNModelWrapper,
    MLPModelWrapper,
    AutoencoderMLPModelWrapper,
    AutoencoderRandomDropoutMLPModelWrapper,
    RandomDropoutLSTMModelWrapper,
    AutoencoderRandomDropoutBidirectionalLSTMModelWrapper,
    AutoencoderRandomDropoutTimeDistributedCNNLSTMModelWrapper,
    AutoencoderRandomDropoutCNNLSTMModelWrapper,
    AutoencoderCNNLSTMTimeDistributedModelWrapper,
    RandomDropoutGRUModelWrapper,
    AutoencoderRandomDropoutCNNModelWrapper,
    AutoencoderRandomDropoutMultiCNNModelWrapper,
    AutoencoderRandomDropoutMLPLSTMModelWrapper,
    AutoencoderRandomDropoutMLPGRUModelWrapper,
    ResNetLSTMModelWrapper,
    ResNetClassificationModelWrapper
]


for model_class in ALL:

    print(str(model_class))

    for i in range(1, 13):
        config.apply_metadata(getattr(Metadata, "version_" + str(i))())
        model = model_class(config)

        #model.train_sequentially = False
        model.fit()
        model.plot_train()
        model.save_train_figure()

        Plotly.ZOOM = -305
        model.plot_predict(425, show_confidence_interval=True)
        model.save_prediction_figure()
        tf.keras.backend.clear_session()


# TF Problem: https://github.com/tensorflow/tensorflow/issues/27511 -> solution: https://github.com/tensorflow/tensorflow/commit/5956c7e9c44e23cd1a006df872ae468201fdb600#diff-e329ed6b8d30dca9a441689005047035

