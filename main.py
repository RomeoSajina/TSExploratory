from sklearn.exceptions import DataConversionWarning
import tensorflow as tf
import warnings
import datetime
from core.__init__ import *
from core.cddm import *
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

config = DataFactory.load_ts()

#config.apply_metadata(Metadata.version_1())
#config.apply_metadata(Metadata.version_2())
#config.apply_metadata(Metadata.version_3())
config.apply_metadata(Metadata.version_4())


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
model = AutoencoderCNNModelWrapper(config)
model = AutoencoderMultiCNNModelWrapper(config)
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

#model.train_sequentially = False

model.fit()

model.plot_train()
#model.save_train_figure()

model.plot_predict_multiple()
model.plot_predict()

Plotly.ZOOM = -50
model.plot_predict(425)
#model.save_prediction_figure()

model.plot_multiple_and_train()
Plotly.savefigpkg(model.info() + ".svg")


for i in range(1, 13):
    config.apply_metadata(getattr(Metadata, "version_"+str(i))())
    model = AutoencoderCNNModelWrapper(config)
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
    AutoencoderCNNModelWrapper,
    AutoencoderMultiCNNModelWrapper,
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
        #import gc
        #gc.collect() ???


# TF Problem: https://github.com/tensorflow/tensorflow/issues/27511 -> solution: https://github.com/tensorflow/tensorflow/commit/5956c7e9c44e23cd1a006df872ae468201fdb600#diff-e329ed6b8d30dca9a441689005047035



## Concept-drift

def load_for_concept_drift():

    y_pred_range = 60
    config = DataFactory.load_ts(end_date=datetime.datetime(2017, 7, 7))
    config.gaps = [y_pred_range for x in reversed(range(1, 7))]
    config.apply_metadata(Metadata.version_2())

    model = AutoencoderMLPModelWrapper(config)
    model._build_model_file_name = lambda: config.base_dir + "models/" + model.info() + "_test.h5"
    model.save = lambda: model.model.save(model._build_model_file_name())

    model.epochs = 2000
    model.refitting_lr = 0.0001
    model.fit()

    return config, model

config, model = load_for_concept_drift()

#model.epochs = 1000

config.cddm = AdaptiveCDDM()

config.cddm = KruskalWallisInputCDDM()
config.cddm = KruskalWallisOutputCDDM()
config.cddm = CUSUMCDDM()
config.cddm = BFASTCDDM()
config.cddm = IgnoreCDDM()

Plotly.ZOOM = -10
model.plot_predict_multiple2()

config.reset()








########### IMAGES ###########

Plotly.set_figure_size((18, 10))
Plotly.ALPHA = .8
# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
plt.rcParams.update({'font.size': 16})


Plotly.plot_ts(config)
Plotly.savefigpkg("info/ts.svg")


Plotly.plot_seasonal_decompose(config)
Plotly.savefigpkg("info/decompose.svg")


Plotly.plot_seasonal_decompose(config, "trend")
Plotly.savefigpkg("info/trend.svg")


"""size = Plotly.FIGURE_SIZE
Plotly.set_figure_size((18, 6.5))
plt.subplot(122)
DataFactory.plot_yearly()
plt.subplot(121)
Plotly.plot_ts(config, True)
plt.subplots_adjust(top=.95, bottom=0.08, left=0.05, right=0.95, hspace=0.25, wspace=0.2)
Plotly.set_figure_size(size)"""
DataFactory.plot_yearly()
Plotly.show()
Plotly.savefigpkg("info/seasonality.svg")


Plotly.plot_acf(config)
Plotly.savefigpkg("info/acf.svg")


Plotly.plot_pacf(config)
Plotly.savefigpkg("info/pacf.svg")


plt.plot(config.all.y.diff())
plt.xlabel("Datum")
plt.title("Diferencirani vremenski niz rezervacija")
Plotly.show()
Plotly.savefigpkg("info/diff.svg")


BFASTCDDM.plot_bfast(config.all, datetime.datetime(2017, 7, 7))
Plotly.show()
Plotly.savefigpkg("info/bfast.svg")



ALL = [
    PersistentModelWrapper,
    SeasonalPersistentModelWrapper,

    ARModelWrapper,
    MAModelWrapper,
    ARMAModelWrapper,
    ARIMAModelWrapper,

    ProphetModelWrapper,

    [MLPModelWrapper, 8],
    [GRUModelWrapper, 11],
    [LSTMModelWrapper, 6],
    [CNNModelWrapper, 3],
    [MultiCNNModelWrapper, 3],
    [BidirectionalLSTMModelWrapper, 4],
    [AutoencoderMLPModelWrapper, 12],
    [AutoencoderCNNModelWrapper, 7],
    [AutoencoderMultiCNNModelWrapper, 3]
]
    #SARIMAXModelWrapper,


#config.apply_metadata(Metadata.version_4())
for model_class in ALL:

    model_ver = 1

    if isinstance(model_class, list):
        model_ver = model_class[1]
        model_class = model_class[0]

    print(str(model_class), str(model_ver))

    config.apply_metadata(getattr(Metadata, "version_" + str(model_ver))())
    model = model_class(config)

    model.fit()

    Plotly.ZOOM = -50
    #model.plot_multiple_and_train()
    model.plot_predict_multiple()
    Plotly.savefigpkg("model/" +model.info() + ".svg")


# CD

config, model = load_for_concept_drift()
config.cddm = IgnoreCDDM()
Plotly.ZOOM = -10
model.plot_predict_multiple2()
Plotly.savefigpkg("cddm/IgnoreCDDM.svg")


config, model = load_for_concept_drift()
config.cddm = AdaptiveCDDM()
Plotly.ZOOM = -10
model.plot_predict_multiple2()
Plotly.savefigpkg("cddm/AdaptiveCDDM.svg")


config, model = load_for_concept_drift()
config.cddm = KruskalWallisInputCDDM()
Plotly.ZOOM = -10
model.plot_predict_multiple2()
Plotly.savefigpkg("cddm/KruskalWallisInputCDDM.svg")
# False, True, True, True, False


config, model = load_for_concept_drift()
config.cddm = KruskalWallisOutputCDDM()
Plotly.ZOOM = -10
model.plot_predict_multiple2()
Plotly.savefigpkg("cddm/KruskalWallisOutputCDDM.svg")
# True, True, True, True, False

config, model = load_for_concept_drift()
config.cddm = CUSUMCDDM()
Plotly.ZOOM = -10
model.plot_predict_multiple2()
Plotly.savefigpkg("cddm/CUSUMCDDM.svg")
# False, True, False, True, True


config, model = load_for_concept_drift()
config.cddm = BFASTCDDM()
Plotly.ZOOM = -10
model.plot_predict_multiple2()
Plotly.savefigpkg("cddm/BFASTCDDM.svg")
# False...







#### Activation fuctions
# https://github.com/mnielsen/neural-networks-and-deep-learning
size = Plotly.FIGURE_SIZE
Plotly.set_figure_size((18, 4))

z = np.arange(-2, 2, .1)
zero = np.zeros(len(z))
y = np.max([zero, z], axis=0)
plt.subplot(141)
plt.plot(z, y)
plt.gca().set_ylim([-2.0, 2.0])
plt.gca().set_xlim([-2.0, 2.0])
plt.grid(True)
plt.title('ReLu')


z = np.arange(-5, 5, .1)
t = np.tanh(z)
plt.subplot(142)
plt.plot(z, t)
plt.gca().set_ylim([-1.0, 1.0])
plt.gca().set_xlim([-5, 5])
plt.grid(True)
plt.title("Tanh")


z = np.arange(-5, 5, .1)
sigma_fn = np.vectorize(lambda z: 1/(1+np.exp(-z)))
sigma = sigma_fn(z)
plt.subplot(143)
plt.plot(z, sigma)
plt.gca().set_ylim([-0.5, 1.5])
plt.gca().set_xlim([-5, 5])
plt.grid(True)
plt.title('Sigmoid')


z = np.arange(-5, 5, .02)
step_fn = np.vectorize(lambda z: 1.0 if z >= 0.0 else 0.0)
step = step_fn(z)
plt.subplot(144)
plt.plot(z, step)
plt.gca().set_ylim([-0.5, 1.5])
plt.gca().set_xlim([-5, 5])
plt.grid(True)
plt.title('Step')

Plotly.show()
Plotly.savefigpkg("info/activation_fnc.svg")
Plotly.set_figure_size(size)





##### STATS ######

from core.model.nn_model import NNBaseModelWrapper

for model_class in ALL:

    print(str(model_class))

    end = 13 if issubclass(model_class, NNBaseModelWrapper) else 2

    for i in range(1, end):
        config.apply_metadata(getattr(Metadata, "version_" + str(i))())
        model = model_class(config)

        model.fit()

        StatsCollector.collect_mtpl_model_stats(config, model)

        tf.keras.backend.clear_session()
        import gc
        gc.collect()
