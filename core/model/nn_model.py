import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, Flatten, MaxPooling1D, Conv1D, ConvLSTM2D, Layer, BatchNormalization, RepeatVector
from tensorflow.python.keras.layers import Dense, Dropout, Input, Add, Activation, ZeroPadding1D, AveragePooling1D
from tensorflow.python.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.utils import np_utils
from core import Config
from core import Plotly
from core.model.nn import Properties
from core.model.base_model import BaseModelWrapper
from core.model.nn.callback import EarlyStoppingAtMinLoss
from core.model.nn.layer import RandomDropout
from core.data_factory import DataFactory
import pandas as pd
import math
import numpy as np
import scipy.stats
import random as rn
from abc import abstractmethod
import datetime


class NNBaseModelWrapper(BaseModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self._train_sequentially = True
        self._last_end_date = config.end_date
        self.version = config.version
        self.optimizer = "adam"
        self.refitting_lr = 0.000001
        self.loss_metric = "mse"
        self.epochs = 1000
        self.steps_per_epoch = None
        self.callbacks = []
        self._model_loss = np.Inf
        self.__max_y_limit = max(config.train.y.values) * 1.2 # aprove 20% more than max
        self.batch_size = config.n_steps + 1
        self._history = None
        self.last_epoch = 0
        self.include_tensorboard_callback = False # CMD: tensorboard --logdir logs

        self.refresh_properties()

    def refresh_properties(self):
        Properties.load(self)
        #self.init_callbacks()

    @property
    def train_sequentially(self):
        return self._train_sequentially

    @train_sequentially.setter
    def train_sequentially(self, value):
        self._train_sequentially = value
        self.refresh_properties()

    @property
    def model_loss(self):
        return self._model_loss

    @model_loss.setter
    def model_loss(self, value):
        self._model_loss = value

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_steps, self.config.n_features))

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], x.shape[1], self.config.n_features))

    @abstractmethod
    def create_model(self):
        pass

    def init_callbacks(self):
        callbacks = [
            EarlyStoppingAtMinLoss(monitor='loss',
                                   patience=50,
                                   min_delta=0,  # 1e-5,
                                   verbose=self.config.verbose,
                                   restore_best_weights=True,
                                   margin_loss=self.config.margin_loss,
                                   baseline=self._model_loss)
        ]

        if self.include_tensorboard_callback:
            callbacks.append(tf.keras.callbacks.TensorBoard())

        self.callbacks = callbacks

    def fit_model(self, refitting=False):

        if refitting:
            self._retrain()
        else:
            self._train()

    def predict(self, days=None, return_confidence_interval=False):

        if not return_confidence_interval:
            return self._predict(days)

        elif not self.__contains_random_dropout():
            return self._predict(days), None

        else:
            vals = []

            for r in range(100):
                vals.append(self._predict(days))

            means = np.mean(vals, axis=0)

            """
            mn, mx = np.min(vals, axis=0), np.max(vals, axis=0)

            if (mn == mx).all():
                conf_int = None
            else:
                conf_int = np.array([mn * (2 - self._confidence), mx * self._confidence]).T[0]
            """
            # TODO: fix
            def calc_conf_int(data, confidence):
                m = scipy.mean(data)
                h = scipy.stats.sem(data) * scipy.stats.t.ppf((1 + confidence) / 2, len(data) - 1)
                return m - h,  m + h

            vals = np.array(vals)
            conf_int = list()
            for i in range(vals.shape[1]):
                _min, _max = calc_conf_int(vals[:, i, :].flatten(), self._confidence)
                conf_int.append([_min, _max])

            return means, np.array(conf_int)

    def _predict(self, days=None):
        # Recursive Multi-step Forecast

        c = self.config
        train, scaler, n_steps, n_features, n_seq, target_date, end_date = \
            c.train, c.scaler, c.n_steps, c.n_features, c.n_seq, c.target_date, c.end_date

        if days is None:
            days = (target_date - end_date).days + 1

        if scaler is None:
            values = train.y.values
        else:
            values = scaler.fit_transform(np.array(train.y.values).reshape(-1, 1))

        predictions = list()
        while len(predictions) < days: #for i in range(days):

            start_idx = len(train) - n_steps + len(predictions)
            end_idx = len(train)

            if start_idx < end_idx:
                x_input = np.append(np.array(values[start_idx:end_idx]), np.array(predictions))
            else:
                x_input = np.array(predictions[len(predictions) - n_steps:len(predictions)])

            x_input = self._reshape_predict_input(x_input)

            yhat = self.model.predict(x_input, verbose=0)

            predictions.extend(self._format_predicted_values(yhat[0]))
            #predictions.append(self.__format_predicted_value(yhat[0][0]))

        if scaler is not None:
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # return only first N requested days
        return predictions[:days]

    def predict_on_train(self):

        c = self.config
        train, scaler, n_steps, n_features, n_seq = c.train, c.scaler, c.n_steps, c.n_features, c.n_seq

        if scaler is None:
            values = train.y.values
        else:
            values = scaler.fit_transform(np.array(train.y.values).reshape(-1, 1))

        days = len(values)
        predictions = list()
        predictions_multistep = list()

        """
            HOW TO: take n_steps starting from begining of train set, make prediction and increment by one
            PSEUDO:
                    while len(predictions) < len(train) - n_steps # Because we cant predict first n_steps
                        
                        x_input = next_step_dataset(train, index)
                        
                        predictions.append(model.predict(x_input))
        """

        while len(predictions) < days - n_steps: # for i in range(len(train) - n_steps):

            # One-step Forecast
            start_idx = len(predictions)
            end_idx = start_idx + n_steps

            x_input = np.array(values[start_idx:end_idx])

            # Recursive Multi-step Forecast
            if len(predictions_multistep) < n_steps: # If you dont have enough data use training
                x_input_ms = np.append(np.array(values[len(predictions_multistep):n_steps]), np.array(predictions_multistep))
            else: # use only predicted to predict new one
                x_input_ms = np.array(predictions_multistep[len(predictions_multistep) - n_steps:len(predictions_multistep)])

            #print(x_input.shape, x_input_ms.shape)

            x_input = self._reshape_predict_input(x_input)
            x_input_ms = self._reshape_predict_input(x_input_ms)

            #print(x_input.shape, x_input_ms.shape)

            yhat = self.model.predict(x_input, verbose=0)
            yhat_ms = self.model.predict(x_input_ms, verbose=0)

            predictions.extend(self._format_predicted_values(yhat[0]))
            predictions_multistep.extend(self._format_predicted_values(yhat_ms[0]))

            #print(len(predictions), len(predictions_multistep))

        if scaler is not None:
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            predictions_multistep = scaler.inverse_transform(np.array(predictions_multistep).reshape(-1, 1))

        # return only first N requested days
        return predictions[:days], predictions_multistep[:days]

    def _format_predicted_values(self, vals):

        assert isinstance(vals, (list, tuple, np.ndarray)), "Values for formating are not in shape of array"

        vals = np.array(vals).flatten()

        fmt_vals = list()

        for v in vals:
            fmt_vals.append(self.__format_predicted_value(v))

        return fmt_vals

    def __format_predicted_value(self, val):

        assert not isinstance(val, (list, tuple, np.ndarray)), "Value for formating is in shape of array"

        lower_limit = 0 # TODO: scalar.min if not null

        y = lower_limit if math.isnan(val) or val < lower_limit else val
        yt = y

        if self.config.scaler is not None:
            upper_limit = 1 # TODO: scaler.max
            y = y if y <= upper_limit and not math.isinf(y) else upper_limit
            yt = self.config.scaler.inverse_transform(np.array(y).reshape(-1, 1))[0][0]

        if math.isnan(yt) or yt < 0:
            y = 0

        elif yt > self.__max_y_limit:
            y = self.__max_y_limit

            if self.config.scaler is not None:
                y = self.config.scaler.transform(np.array(y).reshape(-1, 1))[0][0]

        if val != y and self.config.verbose > 1:
            print("Transformed: " + str(val) + " -> " + str(y))

        return y

    def _train(self):

        self.init_callbacks()

        if self.train_sequentially:

            X, y = self.X_y()

            # with tf.device("/device:CPU:0"):
            self._history = self.model.fit(X, y,
                                           epochs=self.epochs,
                                           verbose=self.config.verbose,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           steps_per_epoch=self.steps_per_epoch,
                                           callbacks=self.callbacks)

        else:

            yearly = DataFactory.decompose_by_year(self.config.train, self.config.target_date)

            last_year = None

            for y in yearly:

                # Create a dummy config, needed for X_y method (only 'y' in train set is mandatory)
                dummy_config = self.config.copy()
                dummy_config.train = pd.DataFrame(y["y"], columns=["y"], index=y["x"])

                if last_year is not None:
                    # If last year is available, take last n_steps of data from it to be able to predict first of this year
                    n_steps_last_year = pd.DataFrame(last_year["y"][-self.config.n_steps:], columns=["y"], index=last_year["x"][-self.config.n_steps:])
                    dummy_config.train = pd.concat([n_steps_last_year, dummy_config.train])

                last_year = y

                if self.config.verbose > 0:
                    print("Training model for year: " + str(y["year"]))
                    print("Train shape: " + str(dummy_config.train.shape))

                X, y = self.X_y(dummy_config)

                # TODO: adjust learning rate all times except first time if it doesn't do on it's own already
                #K.set_value(self.model.optimizer.lr, self.refitting_lr)

                history = self.model.fit(X, y,
                                         epochs=self.epochs,
                                         verbose=self.config.verbose,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         steps_per_epoch=self.steps_per_epoch,
                                         callbacks=self.callbacks)

                # TODO: init(_history=None) then if _history is None: _history = history else: merge(_history, history)
                self._history = history

        early_stoping = self.callbacks[0]
        early_stoping.baseline = early_stoping.best
        self._model_loss = early_stoping.best
        self.last_epoch = early_stoping.last_epoch
        #Properties.save(self)

    def _retrain(self):

        self.init_callbacks()

        dummy_config = self.config.copy()

        #if self.config.verbose > 0:
        #    print("Refitting NN model. End_date: " + self.config.end_date.strftime("%d.%m.%Y") + ", size of values after it: " + str(len(self.config.train[self.config.end_date:])))

        #previous_gap = 60  # TODO:
        #new_min_date = self.config.end_date - datetime.timedelta(days=self.config.n_steps + previous_gap)

        new_min_date = self.config.min_date - datetime.timedelta(days=self.config.n_steps + 365) # TODO:....

        # TODO: filtrirat samo najnovije vrijednosti koje model još nije vidio?
        # npr. kada je gap iz 60 u 30 uzmi tih 30 dana razlike,
        # ali onda kad je iz 30 u 7 dana uzmi samo tih 23 dana, a ne svih 53
        dummy_config.train = dummy_config.train[new_min_date:]

        if self.config.verbose > 0:
           print("Refitting NN model. min_date: " + self.config.min_date.strftime("%d.%m.%Y") + ", size of values after it: " + str(len(dummy_config.train[self.config.min_date:])))

        #if self.config.verbose > 0:
        #    print("New calculated min training date is: " + new_min_date.strftime("%d.%m.%Y") + ", size of new values is: " + str(len(dummy_config.train)))

        X, y = self.X_y(dummy_config)

        # Decrease the learning rate
        K.set_value(self.model.optimizer.lr, self.refitting_lr)
        self.model.fit(X, y,
                       epochs=int(self.epochs / 5),
                       verbose=0,#self.config.verbose,
                       batch_size=self.batch_size,
                       callbacks=self.callbacks,
                       steps_per_epoch=self.steps_per_epoch,
                       shuffle=False)

        # In a perfect world we would save latest model here
        # model.save_latest()

    def load(self):
        #custom_objects={"tf": tf}
        self.model = load_model(self._build_model_file_name(), custom_objects={"RandomDropout": RandomDropout})

    def save(self):
        Properties.save(self)
        self.model.save(self._build_model_file_name())

    def n_params(self):
        return 0 if self.model is None else self.model.count_params()

    def info(self):
        return self.__class__.__name__.replace("ModelWrapper", "") + "_seq_" + str(self.train_sequentially).lower() + "_ver_" + str(self.version)

    def _build_model_file_name(self):
        name = self.config.base_dir + "models/final/" + self.__class__.__name__.replace("ModelWrapper", "")
        name += "_seq_" + str(self.train_sequentially).lower() + "_ver_" + str(self.version) + ".h5"
        return name

    def save_train_figure(self):
        self.__save_figure("train")

    def save_prediction_figure(self):
        self.__save_figure("prediction")

    def __save_figure(self, info):
        model_info = self.__class__.__name__.replace("ModelWrapper", "") + "_seq_" + str(self.train_sequentially).lower() + "_ver_" + str(self.version) + "_" + info
        name = self.config.base_dir + "figures/models/" + model_info
        #name += "_seq_" + str(self.train_sequentially).lower() + "_ver_" + str(self.version) + "_" + info
        Plotly.savefig(name + ".png", model_info)

    @staticmethod
    def split_sequence(sequence, n_steps, n_outputs):
        # split a univariate sequence into samples

        assert not np.any(np.isnan(sequence)), "Sequence contains nan values"

        X, y = list(), list()

        for i in range(len(sequence)):

            end_ix = i + n_steps
            out_end_ix = end_ix + n_outputs

            if out_end_ix > len(sequence) - 1:
                break

            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y).reshape(len(y), n_outputs)

    def X_y(self, config: Config = None):

        if config is None:
            c = self.config
        else:
            c = config

        train, scaler, n_steps, n_outputs = c.train, c.scaler, c.n_steps, c.n_outputs

        values = train.y.values

        if scaler is not None:
            values = scaler.fit_transform(np.array(values).reshape(-1, 1))

        X, y = NNBaseModelWrapper.split_sequence(values, n_steps=n_steps, n_outputs=n_outputs)

        X = self._reshape_train_input(X)

        # Set seed
        np.random.seed(1234)
        rn.seed(1234)
        #tf.random.set_seed(1234) #Colab Error: module 'tensorflow._api.v1.random' has no attribute 'set_seed'

        return X, y

    def summary(self, print_shapes=False):

        if self.model is not None:
            self.model.summary()

            if print_shapes:
                print("Name:".ljust(20), "Input shape:".ljust(20), "Output shape:".ljust(20))
                for l in self.model.layers:
                    print(l.name.ljust(20), str(l.input_shape).ljust(20), str(l.output_shape).ljust(20))

    """
    def plot_predict(self, days=None, show_confidence_interval=True):
        #predictions = self.predict(days=days)
        #Plotly.plot_predictions(config=self.config, predictions=predictions)
        vals = []

        for r in range(100):
            vals.append(self.predict(days))

        mean = np.mean(vals, axis=0)
        mn, mx = np.min(vals, axis=0), np.max(vals, axis=0)

        if (mn == mx).all():
            conf_int = None
        else:
            conf_int = np.array([mn * (2-self._confidence), mx * self._confidence]).T[0]

        Plotly.plot_predictions(config=self.config, predictions=mean, conf_int=conf_int)
    """
    def __contains_random_dropout(self):
        return np.array([isinstance(l, RandomDropout) for l in self.model.layers]).any()

    def plot_history(self):

        if self._history is None:
            print("History is empty.")
        else:
            Plotly.plot_nn_train_history(self._history)


class CustomModelWrapper(NNBaseModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self.save_load_model = False
        self.reshape_train_input_fnc = None
        self.reshape_predict_input_fnc = None
        self.create_model()

    def create_model(self):
        self.model = Sequential()

    def fit(self):
        self.fit_model()

    def add(self, layer):
        assert isinstance(layer, Layer), "layer must be an instance of Layer"

        self.model.add(layer)

    def _train(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_metric)
        super()._train()

    def _reshape_train_input(self, x):

        if self.reshape_train_input_fnc is not None:
            return self.reshape_train_input_fnc(x)

        return super()._reshape_train_input(x)

    def _reshape_predict_input(self, x):

        if self.reshape_predict_input_fnc is not None:
            return self.reshape_predict_input_fnc(x)

        return super()._reshape_predict_input(x)


class MLPModelWrapper(NNBaseModelWrapper):
    """
        Done!
    """
    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_steps))

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_steps))

    def create_model(self):

        model = Sequential()
        model.add(Dense(500, activation='relu', input_dim=self.config.n_steps))
        model.add(Dropout(0.2))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class AutoencoderMLPModelWrapper(MLPModelWrapper):
    """
        Autoencoder - decoder MLP
        Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(Dense(500, activation='relu', input_dim=self.config.n_steps))
        model.add(Dropout(0.2))
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class AutoencoderRandomDropoutMLPModelWrapper(MLPModelWrapper):
    """
        Done!
        Zanimljiv rezultat n_output=7
    """
    def create_model(self):

        model = Sequential()
        model.add(Dense(500, activation='relu', input_dim=self.config.n_steps))
        model.add(RandomDropout())
        model.add(Dense(300, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(100, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(300, activation='relu'))
        model.add(RandomDropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class AutoencoderRandomDropoutMLPLSTMModelWrapper(NNBaseModelWrapper):
    """
        Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(Dense(500, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(RandomDropout())
        model.add(Dense(300, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(100, activation='relu'))
        model.add(RandomDropout())
        model.add(LSTM(100, activation="relu", return_sequences=True))
        model.add(Dense(300, activation='relu'))
        model.add(RandomDropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(RandomDropout())
        model.add(Flatten())
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_steps, self.config.n_features))

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], x.shape[1], self.config.n_features))


class AutoencoderRandomDropoutMLPGRUModelWrapper(AutoencoderRandomDropoutMLPLSTMModelWrapper):
    """
        Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(Dense(500, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(RandomDropout())
        model.add(Dense(300, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(100, activation='relu'))
        model.add(RandomDropout())
        model.add(GRU(100, activation="relu", return_sequences=True))
        model.add(Dense(300, activation='relu'))
        model.add(RandomDropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(RandomDropout())
        model.add(Flatten())
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class LSTMModelWrapper(NNBaseModelWrapper):
    """
    Done!
    """
    def __init__(self, config: Config):
        super(LSTMModelWrapper, self).__init__(config)
        self.optimizer = "nadam"

    def create_model(self):

        model = Sequential()
        model.add(LSTM(300, activation='relu', return_sequences=True, input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(300, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class RandomDropoutLSTMModelWrapper(LSTMModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(LSTM(300, activation='relu', return_sequences=True, input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(RandomDropout())
        model.add(LSTM(300, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(self.config.n_outputs))
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class BidirectionalLSTMModelWrapper(NNBaseModelWrapper):
    """
    Done! los model
    """
    def create_model(self):

        model = Sequential()
        model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True), input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(100, activation='relu')))
        model.add(Dropout(0.2))
        model.add(Dense(300, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class AutoencoderRandomDropoutBidirectionalLSTMModelWrapper(BidirectionalLSTMModelWrapper):
    """
    Done! los model
    """
    def create_model(self):

        model = Sequential()
        model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True), input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(RandomDropout())
        model.add(Bidirectional(LSTM(100, activation='relu')))
        model.add(RandomDropout())
        model.add(Dense(100, activation="relu"))
        model.add(RandomDropout())
        model.add(Dense(500, activation="relu"))
        model.add(RandomDropout())
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class TimeDistributedCNNLSTMModelWrapper(NNBaseModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=7, activation='relu'), input_shape=(None, self.config.n_steps, self.config.n_features)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(Dropout(0.2))
        model.add(LSTM(300, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_seq, self.config.n_steps, self.config.n_features))

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_seq, self.config.n_steps, self.config.n_features))


class AutoencoderRandomDropoutTimeDistributedCNNLSTMModelWrapper(TimeDistributedCNNLSTMModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=7, activation='relu'), input_shape=(None, self.config.n_steps, self.config.n_features)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(RandomDropout())
        model.add(LSTM(300, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(100, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(500, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(self.config.n_outputs))
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class CNNLSTMModelWrapper(NNBaseModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(ConvLSTM2D(return_sequences=True,
                             filters=64,
                             kernel_size=(1, 5),
                             activation='relu',
                             input_shape=(self.config.n_seq, 1, self.config.n_steps, self.config.n_features)))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3), return_sequences=True, activation="relu"))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_seq, 1, self.config.n_steps, self.config.n_features))

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_seq, 1, self.config.n_steps, self.config.n_features))


class AutoencoderRandomDropoutCNNLSTMModelWrapper(CNNLSTMModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()

        model.add(ConvLSTM2D(return_sequences=True,
                             filters=64,
                             kernel_size=(1, 5),
                             activation='relu',
                             input_shape=(self.config.n_seq, 1, self.config.n_steps, self.config.n_features)))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3), return_sequences=True, activation="relu"))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(500, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class AutoencoderCNNLSTMTimeDistributedModelWrapper(CNNLSTMModelWrapper):
    """
    Done!
    """
    def create_model(self):
        """
        shape: [samples, timesteps, rows, cols, channels]
        TimeDistributed racuna za svaki timestep zasebno, zato je output modela = n_outputs
        """
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(self.config.n_seq, 1, self.config.n_steps, self.config.n_features)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(RepeatVector(self.config.n_outputs))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(RandomDropout())
        model.add(TimeDistributed(Dense(300, activation='relu')))
        model.add(RandomDropout())
        model.add(TimeDistributed(Dense(1)))
        model.add(Flatten())

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class GRUModelWrapper(NNBaseModelWrapper):
    """
    Done!
    """
    def __init__(self, config: Config):
        super(GRUModelWrapper, self).__init__(config)
        self.optimizer = "nadam"

    def create_model(self):

        model = Sequential()
        model.add(GRU(300, return_sequences=True, input_shape=(self.config.n_steps, self.config.n_features), activation='relu'))
        model.add(Dropout(0.2))
        model.add(GRU(300, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))
        # model.compile(optimizer=SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class RandomDropoutGRUModelWrapper(GRUModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(GRU(300, return_sequences=True, input_shape=(self.config.n_steps, self.config.n_features), activation='relu'))
        model.add(RandomDropout())
        model.add(GRU(300, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(self.config.n_outputs))
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class CNNModelWrapper(NNBaseModelWrapper):
    """
    Done!
    """
    def create_model(self):
        """
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        # model.add(Dense(100, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.config.n_outputs))
        """

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_steps, self.config.n_features))

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_steps, self.config.n_features))


class AutoencoderCNNModelWrapper(CNNModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()

        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class AutoencoderRandomDropoutCNNModelWrapper(CNNModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()

        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(RandomDropout())
        model.add(Dense(100, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(500, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class MultiCNNModelWrapper(NNBaseModelWrapper):
    """
    Done!
    """
    def create_model(self):
        """
        model.add(Conv1D(filters=64, kernel_size=15, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))
        """

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=15, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_steps, self.config.n_features))

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_steps, self.config.n_features))


class AutoencoderMultiCNNModelWrapper(MultiCNNModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=15, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class AutoencoderRandomDropoutMultiCNNModelWrapper(MultiCNNModelWrapper):
    """
    Done!
    """
    def create_model(self):

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=15, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(RandomDropout())
        model.add(Dense(100, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(500, activation='relu'))
        model.add(RandomDropout())
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

"""
class ResNet50ModelWrapper(NNBaseModelWrapper):

    # Thanks to https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb

    def __init__(self, config: Config):
        config.scaler = None
        super().__init__(config)
        self.n_classes = int(max(config.all.y) * 1.2)
        self.steps_per_epoch = 1

    def _build_identity_block(self, X, f, filters, stage, block):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def _build_convolutional_block(self, X, f, filters, stage, block, s=2):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path
        X = Conv1D(filters=F1, kernel_size=1, strides=s, name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2c')(X)

        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv1D(filters=F3, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=2, name=bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def _build_resnet50(self, input_shape):

        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        # Zero-Padding
        X = ZeroPadding1D()(X_input)

        # Stage 1
        X = Conv1D(filters=64, kernel_size=7, strides=2, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name='bn_conv1')(X)
        X = Activation('relu')(X)

        X = MaxPooling1D(pool_size=3, strides=2)(X)

        # Stage 2
        X = self._build_convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X = self._build_identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self._build_identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3 (≈4 lines)
        X = self._build_convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = self._build_identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self._build_identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self._build_identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 (≈6 lines)
        X = self._build_convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = self._build_identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self._build_identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self._build_identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self._build_identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self._build_identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5 (≈3 lines)
        X = self._build_convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        X = self._build_identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self._build_identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        X = AveragePooling1D(pool_size=2, name="avg_pool")(X)

        # output layer
        X = Flatten()(X)
        X = Dense(self.n_classes, activation='softmax', name='fc' + str(self.n_classes), kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_model(self):
        self.model = self._build_resnet50(input_shape=(self.config.n_steps, self.n_classes))

    def X_y(self, config: Config = None):

        X, y = super().X_y(config)

        X = tf.one_hot(X, self.n_classes, axis=2)

        y = np_utils.to_categorical(y.reshape(y.shape[0]), self.n_classes)

        return X, y

    def _format_predicted_values(self, vals):
        p = np.argmax(vals, axis=0)
        return [p]

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_steps))

    def _reshape_predict_input(self, x):
        x = x.astype(np.int64)
        x = tf.one_hot(x, self.n_classes, axis=1).numpy()
        return x.reshape(1, x.shape[0], x.shape[1])
"""


class ResNetClassificationModelWrapper(NNBaseModelWrapper):

    def __init__(self, config: Config):
        config.scaler = None
        super().__init__(config)
        self.n_classes = int(max(config.all.y) * 1.2)
        self.steps_per_epoch = 1


    def _build_identity_block(self, X, f, filters, stage, block):

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2 = filters

        X_shortcut = X

        X = Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        """
        X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2c')(X)
        """

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def _build_convolutional_block(self, X, f, filters, stage, block, s=2):

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2 = filters

        X_shortcut = X

        X = Conv1D(filters=F1, kernel_size=1, strides=s, name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X_shortcut = Conv1D(filters=F2, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=2, name=bn_name_base + '1')(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def _build_resnet(self, input_shape):

        X_input = Input(input_shape)

        X = ZeroPadding1D()(X_input)

        X = Conv1D(filters=64, kernel_size=7, strides=2, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name='bn_conv1')(X)
        X = Activation('relu')(X)

        X = MaxPooling1D(pool_size=3, strides=2)(X)

        X = self._build_convolutional_block(X, f=3, filters=[64, 64], stage=2, block='a', s=1)
        X = self._build_identity_block(X, 3, [64, 64], stage=2, block='b')

        X = self._build_convolutional_block(X, f=3, filters=[128, 128], stage=3, block='a', s=2)
        X = self._build_identity_block(X, 3, [128, 128], stage=3, block='b')

        X = self._build_convolutional_block(X, f=3, filters=[256, 256], stage=4, block='a', s=2)
        X = self._build_identity_block(X, 3, [256, 256], stage=4, block='b')

        X = self._build_convolutional_block(X, f=3, filters=[512, 512], stage=5, block='a', s=2)
        X = self._build_identity_block(X, 3, [512, 512], stage=5, block='c')

        X = AveragePooling1D(pool_size=2, name="avg_pool")(X)

        X = Flatten()(X)
        X = Dense(self.n_classes, activation='softmax', name='fc' + str(self.n_classes), kernel_initializer=glorot_uniform(seed=0))(X)

        model = Model(inputs=X_input, outputs=X, name='ResNetClassification')

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_model(self):

        self.model = self._build_resnet(input_shape=(self.config.n_steps, self.n_classes))

    def X_y(self, config: Config = None):

        X, y = super().X_y(config)

        X = tf.one_hot(X, self.n_classes, axis=2)

        y = np_utils.to_categorical(y.reshape(y.shape[0]), self.n_classes)

        return X, y

    def _format_predicted_values(self, vals):
        p = np.argmax(vals, axis=0)
        return [p]

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_steps))

    def _reshape_predict_input(self, x):
        x = x.astype(np.int64)
        x = tf.one_hot(x, self.n_classes, axis=1).numpy()
        return x.reshape(1, x.shape[0], x.shape[1])


class ResNetLSTMModelWrapper(NNBaseModelWrapper):

    def __init__(self, config: Config):
        config.scaler = None
        super().__init__(config)
        self.n_classes = int(max(config.all.y))
        self.steps_per_epoch = 1

    def _build_identity_block(self, X, f, filters, stage, block):

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2 = filters

        X_shortcut = X

        X = Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        """
        X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2c')(X)
        """

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def _build_convolutional_block(self, X, f, filters, stage, block, s=2):

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2 = filters

        X_shortcut = X

        X = Conv1D(filters=F1, kernel_size=1, strides=s, name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X_shortcut = Conv1D(filters=F2, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=2, name=bn_name_base + '1')(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def _build_resnet(self, input_shape):

        X_input = Input(input_shape)

        X = ZeroPadding1D()(X_input)

        X = Conv1D(filters=64, kernel_size=7, strides=2, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name='bn_conv1')(X)
        X = Activation('relu')(X)

        X = MaxPooling1D(pool_size=3, strides=2)(X)

        X = self._build_convolutional_block(X, f=3, filters=[64, 64], stage=2, block='a', s=1)
        X = self._build_identity_block(X, 3, [64, 64], stage=2, block='b')

        X = self._build_convolutional_block(X, f=3, filters=[128, 128], stage=3, block='a', s=2)
        X = self._build_identity_block(X, 3, [128, 128], stage=3, block='b')

        X = self._build_convolutional_block(X, f=3, filters=[256, 256], stage=4, block='a', s=2)
        X = self._build_identity_block(X, 3, [256, 256], stage=4, block='b')

        X = self._build_convolutional_block(X, f=3, filters=[512, 512], stage=5, block='a', s=2)
        X = self._build_identity_block(X, 3, [512, 512], stage=5, block='c')

        X = AveragePooling1D(pool_size=2, name="avg_pool")(X)

        X = LSTM(512, activation='relu')(X)
        X = Dense(self.config.n_outputs)(X)

        model = Model(inputs=X_input, outputs=X, name='ResNetLSTM')

        model.compile(optimizer='adam', loss='mse')

        return model

    def create_model(self):
        self.model = self._build_resnet(input_shape=(self.config.n_steps, self.n_classes))

    def X_y(self, config: Config = None):

        X, y = super().X_y(config)

        X = tf.one_hot(X, self.n_classes, axis=2)

        return X, y

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_steps))

    def _reshape_predict_input(self, x):
        x = x.astype(np.int64)
        x = tf.one_hot(x, self.n_classes, axis=1).numpy()
        return x.reshape(1, x.shape[0], x.shape[1])