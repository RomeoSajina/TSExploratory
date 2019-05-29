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
#model.save_load_model = False

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


for i in range(1, 13):
    config.apply_metadata(getattr(Metadata, "version_"+str(i))())
    model = AutoencoderCNNLSTMTimeDistributedModelWrapper(config)
    model.epochs = 1500
    model.fit()


ALL = [
    #BidirectionalLSTMModelWrapper,
    #CNNLSTMModelWrapper,
    #GRUModelWrapper,
    #LSTMModelWrapper,
    #TimeDistributedCNNLSTMModelWrapper,
    #CNNModelWrapper,
    #MultiCNNModelWrapper,
    #MLPModelWrapper,
    #AutoencoderMLPModelWrapper,
    #AutoencoderRandomDropoutMLPModelWrapper,
    #RandomDropoutLSTMModelWrapper,
    #AutoencoderRandomDropoutBidirectionalLSTMModelWrapper,
    #AutoencoderRandomDropoutTimeDistributedCNNLSTMModelWrapper,
    #AutoencoderRandomDropoutCNNLSTMModelWrapper,
    AutoencoderCNNLSTMTimeDistributedModelWrapper,
    RandomDropoutGRUModelWrapper,
    AutoencoderRandomDropoutCNNModelWrapper,
    AutoencoderRandomDropoutMultiCNNModelWrapper,
    #AutoencoderRandomDropoutMLPLSTMModelWrapper,
    #AutoencoderRandomDropoutMLPGRUModelWrapper,
    #ResNetClassificationModelWrapper,
    #ResNetLSTMModelWrapper
]

for model_class in ALL:

    print(str(model_class))

    for i in range(1, 13):
        config.apply_metadata(getattr(Metadata, "version_" + str(i))())
        model = model_class(config)
        #model = TimeDistributedCNNLSTMModelWrapper(config)
        # model.epochs = 2000

        # model.train_sequentially = False
        model.fit()
        model.plot_train()
        model.save_train_figure()

        Plotly.ZOOM = -305
        model.plot_predict(425, show_confidence_interval=False)
        model.save_prediction_figure()
        tf.keras.backend.clear_session()



"""
BEST: 

MLP_seq_true_ver_3
MLP_seq_true_ver_7
MLP_seq_true_ver_8
MLP_seq_true_ver_11
MLP_seq_true_ver_12

AutoencoderMLP_seq_true_ver_3
AutoencoderMLP_seq_true_ver_4
AutoencoderMLP_seq_true_ver_7
AutoencoderMLP_seq_true_ver_8
AutoencoderMLP_seq_true_ver_11
AutoencoderMLP_seq_true_ver_12

AutoencoderRandomDropoutMLP_seq_true_ver_3
AutoencoderRandomDropoutMLP_seq_true_ver_4
AutoencoderRandomDropoutMLP_seq_true_ver_7
AutoencoderRandomDropoutMLP_seq_true_ver_8
AutoencoderRandomDropoutMLP_seq_true_ver_12


BidirectionalLSTM_seq_true_ver_1 - prati pattern
BidirectionalLSTM_seq_true_ver_11

CNN_seq_true_ver_8
CNN_seq_true_ver_12

CNNLSTM_seq_true_ver_8
CNNLSTM_seq_true_ver_12

MultiCNN_seq_true_ver_3
MultiCNN_seq_true_ver_4
MultiCNN_seq_true_ver_12

GRU_seq_true_ver_9 ???


TimeDistributedCNNLSTM_seq_true_ver_4 - krivo ali prati pattern

"""