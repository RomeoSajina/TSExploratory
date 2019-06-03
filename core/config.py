from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime
from core.cddm import EarlyConceptDriftDetectionMethod, IgnoreConceptDriftDetectionMethod, AlwaysConceptDriftDetectionMethod
#from tensorflow.python.keras.callbacks import EarlyStopping, TerminateOnNaN
#from core.model.callback import EarlyStoppingAtMinLoss


class Config:

    _base_dir = "./"

    def __init__(self):

        self._all = pd.DataFrame()
        self._train = pd.DataFrame()
        self._test = pd.DataFrame()
        self.target_date = datetime.datetime(2018, 7, 1)
        self._scaler = MinMaxScaler(feature_range=(0, 1))

        """
        Some ideas on the size and nature of this input include:

        All prior days, up to years worth of data.
        The prior seven days.
        The prior two weeks.
        The prior one month.
        The prior one year.
        The prior week and the week to be predicted from one year ago.
        """
        self.n_steps = 30
        self.n_outputs = 1

        self.n_features = 1
        # Za CNN
        self.n_seq = 1 # ne dela mi veci broj
        self.end_date = datetime.datetime(2015, 5, 1)
        self.min_date = datetime.datetime(2018, 5, 1)
        self.gaps = [60, 30, 7]
        self.cddm = None #AlwaysConceptDriftDetectionMethod()# IgnoreConceptDriftDetectionMethod()# EarlyConceptDriftDetectionMethod()
        self.verbose = 2

        """
        EarlyStopping(monitor='loss',
                      patience=200,
                      min_delta=0,
                      verbose=self.verbose,
                      restore_best_weights=True),
        """
        """
        self.nn_fit_callbacks = [
            EarlyStoppingAtMinLoss(monitor='loss',
                                   patience=100,
                                   min_delta=0,
                                   verbose=self.verbose,
                                   restore_best_weights=False,
                                   margin_loss=5e-6),
            TerminateOnNaN()]
        """
        self.base_dir = Config.base_dir()
        self.version = 1
        self.margin_loss = 5e-4
        self.apply_metadata(Metadata.version_1())

    @staticmethod
    def base_dir():
        return Config._base_dir

    @staticmethod
    def set_base_dir(value):
        Config._base_dir = value

    @staticmethod
    def create(train, test, target_date, scaler, n_steps, n_features, n_seq):
        conf = Config()
        conf._train = train
        conf._test = test
        conf._all = pd.concat([train, test])
        conf.target_date = target_date
        conf._scaler = scaler
        conf.n_steps = n_steps
        conf.n_features = n_features
        conf.n_seq = n_seq
        conf.end_date = train.index[-1]
        conf.min_date = train.index[1]

    @staticmethod
    def build(ts, min_date, end_date, target_date):

        conf = Config()
        conf._all = ts
        conf._train = ts[:end_date]
        conf._test = ts[end_date:]
        conf.target_date = target_date
        conf.end_date = end_date
        conf.min_date = min_date

        return conf

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    @property
    def all(self):
        return self._all

    @property
    def train_and_test(self):
        return self._train, self._test

    def set_train_and_test(self, train, test):
        self._train = train
        self._test = test
        self._all = pd.concat([train, test])

    """
    @property
    def nn_fit_callbacks(self):
        callbacks = [
            EarlyStoppingAtMinLoss(monitor='loss',
                                   patience=50,
                                   min_delta=0, #1e-5,
                                   verbose=self.verbose,
                                   restore_best_weights=True,
                                   margin_loss=5e-4),
            TerminateOnNaN()]

        return callbacks
    """
    def concept_is_drifting(self, new_values: pd.DataFrame, predictions: list):

        # 'new_values' are from test set and not contained in train set,
        # also 'predictions' are made by model that is trained on data that doesn't contains 'new_values'
        is_drifting = self.cddm is not None and self.cddm.is_drifting(self.train, new_values, predictions)

        if self.cddm is not None and self.verbose > 0:
            print("Drifting detected: " + str(is_drifting))

        return is_drifting

    def copy(self):
        newC = Config()
        newC._all = self._all.copy()
        newC._train = self._train.copy()
        newC._test = self._test.copy()
        newC.target_date = self.target_date
        newC._scaler = self._scaler
        newC.n_steps = self.n_steps
        newC.n_features = self.n_features
        newC.n_seq = self.n_seq
        newC.end_date = self.end_date
        newC.min_date = self.min_date
        newC.cddm = self.cddm
        #newC.nn_fit_callbacks = self.nn_fit_callbacks
        newC.verbose = self.verbose
        newC.base_dir = self.base_dir
        newC.n_outputs = self.n_outputs
        newC.version = self.version
        newC.margin_loss = self.margin_loss

        return newC

    def train_and_test_to_csv(self):
        self._train.to_csv(self.base_dir + "data/train.csv")
        self._test.to_csv(self.base_dir + "./data/test.csv")

    def apply_metadata(self, metadata):

        if self.verbose > 1:
            print("Applying Metadata version {0}: \nSteps: {1}\nOutputs: {2}".format(metadata.version, metadata.n_steps, metadata.n_outputs))

        self.n_steps = metadata.n_steps
        self.n_outputs = metadata.n_outputs
        self.version = metadata.version
        self.margin_loss = metadata.margin_loss


# Metadata about diferent versions of models
class Metadata:

    def __init__(self, version, n_steps, n_outputs, margin_loss):
        self.version = version
        self.n_steps = n_steps
        self.n_outputs = n_outputs
        self.margin_loss = margin_loss

    @staticmethod
    def version_1():
        return Metadata(1, 60, 1, 5e-4)

    @staticmethod
    def version_2():
        return Metadata(2, 100, 1, 5e-5)

    @staticmethod
    def version_3():
        return Metadata(3, 200, 1, 3e-5)

    @staticmethod
    def version_4():
        return Metadata(4, 300, 1, 1e-5)

    @staticmethod
    def version_5():
        return Metadata(5, 60, 3, 5e-4)

    @staticmethod
    def version_6():
        return Metadata(6, 100, 3, 5e-5)

    @staticmethod
    def version_7():
        return Metadata(7, 200, 3, 3e-5)

    @staticmethod
    def version_8():
        return Metadata(8, 300, 3, 1e-5)

    @staticmethod
    def version_9():
        return Metadata(9, 60, 7, 5e-4)

    @staticmethod
    def version_10():
        return Metadata(10, 100, 7, 5e-5)

    @staticmethod
    def version_11():
        return Metadata(11, 200, 7, 3e-5)

    @staticmethod
    def version_12():
        return Metadata(12, 300, 7, 1e-5)
