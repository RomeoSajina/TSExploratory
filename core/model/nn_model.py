from core.model.base_model import BaseModelWrapper
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, Flatten, MaxPooling1D, Conv1D, ConvLSTM2D, Layer
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.callbacks import TerminateOnNaN
from core.model.nn.callback import EarlyStoppingAtMinLoss
from core.model.nn.layer import RandomDropout
from core import Config
import pandas as pd
import math
import numpy as np
import random as rn
from core.data_factory import DataFactory
from abc import abstractmethod
import datetime
from core import Plotly


class NNBaseModelWrapper(BaseModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self.train_sequentially = True
        self._last_end_date = config.end_date
        self.version = config.version
        self.optimizer = "adam"
        self.loss_metric = "mse"
        self.epochs = 1000
        self.callbacks = []
        self._model_loss = np.Inf
        self.init_callbacks()
        self.__max_y_limit = max(config.train.y.values) * 1.2 # aprove 20% more than max
        self.batch_size = config.n_steps + 1

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
                                   baseline=self._model_loss),
            TerminateOnNaN()]

        self.callbacks = callbacks

    def fit_model(self, refitting=False):

        if refitting:

            dummy_config = self.config.copy()

            if self.config.verbose > 0:
                print("Refitting NN model. End_date: " +
                      self.config.end_date.strftime("%d.%m.%Y") +
                      ", size of values after it: " + str(len(self.config.train[self.config.end_date:])))

            new_min_date = self.config.end_date - datetime.timedelta(days=self.config.n_steps)

            # TODO: filtrirat samo najnovije vrijednosti koje model još nije vidio?
            # npr. kada je gap iz 60 u 30 uzmi tih 30 dana razlike,
            # ali onda kad je iz 30 u 7 dana uzmi samo tih 23 dana, a ne svih 53
            dummy_config.train = dummy_config.train[new_min_date:]

            if self.config.verbose > 0:
                print("New calculated min training date is: "
                      + new_min_date.strftime("%d.%m.%Y") +
                      ", size of new values is: " + str(len(dummy_config.train)))

            X, y = self.X_y(dummy_config)
            # TODO: dotrenirat model sa early-stoping na greški globalnog modela? Npr. stani kada je greška untuar 20% globalne greške
            self.model.fit(X, y,
                           epochs=self.epochs/5,
                           verbose=self.config.verbose,
                           batch_size=self.batch_size,
                           callbacks=self.callbacks,
                           shuffle=False)
        else:
            self._train()

    def predict(self, days=None):
        return self._predict(days)

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

            predictions.extend(self.__format_predicted_values(yhat[0]))
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
                        
                        predictions.append(x_input)
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

            predictions.extend(self.__format_predicted_values(yhat[0]))
            predictions_multistep.extend(self.__format_predicted_values(yhat_ms[0]))

            #print(len(predictions), len(predictions_multistep))
            #predictions.append(self.__format_predicted_value(yhat[0][0]))
            #predictions_multistep.append(self.__format_predicted_value(yhat_ms[0][0]))

        if scaler is not None:
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            predictions_multistep = scaler.inverse_transform(np.array(predictions_multistep).reshape(-1, 1))

        # return only first N requested days
        return predictions[:days], predictions_multistep[:days]

    def __format_predicted_values(self, vals):

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

        if self.train_sequentially:

            X, y = self.X_y()

            # with tf.device("/device:CPU:0"):
            self.model.fit(X, y,
                           epochs=self.epochs,
                           verbose=self.config.verbose,
                           batch_size=self.batch_size,
                           shuffle=False,
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

                self.model.fit(X, y,
                               epochs=self.epochs,
                               verbose=self.config.verbose,
                               batch_size=self.batch_size,
                               shuffle=False,
                               callbacks=self.callbacks)

        # TODO: ovo ce se pogubit na save/load
        early_stoping = self.callbacks[0]
        early_stoping.baseline = early_stoping.best
        self._model_loss = early_stoping.best

    def load(self):
        self.model = load_model(self._build_model_file_name())

    def save(self):
        self.model.save(self._build_model_file_name())

    def _build_model_file_name(self):
        name = self.config.base_dir + "models/final/" + self.__class__.__name__.replace("ModelWrapper", "")
        name += "_seq_" + str(self.train_sequentially).lower() + "_ver_" + str(self.version) + ".h5"
        return name

    def save_train_figure(self):
        self.__save_figure("train")

    def save_prediction_figure(self):
        self.__save_figure("prediction")

    def __save_figure(self, info):
        name = self.config.base_dir + "figures/" + self.__class__.__name__.replace("ModelWrapper", "")
        name += "_seq_" + str(self.train_sequentially).lower() + "_ver_" + str(self.version) + "_" + info
        Plotly.savefig(name + ".png")
    """
    @staticmethod
    def split_sequence(sequence, n_steps, n_outputs):
        # split a univariate sequence into samples

        X, y = list(), list()

        for i in range(len(sequence)):

            end_ix = i + n_steps

            if end_ix > len(sequence) - 1:
                break

            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)
    """
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

        #c = self.config
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

    def plot_predict(self, days=None):
        #predictions = self.predict(days=days)
        #Plotly.plot_predictions(config=self.config, predictions=predictions)
        vals = []

        for r in range(100):
            vals.append(self.predict(days))

        mean = np.mean(vals, axis=0)
        mn, mx = np.min(vals, axis=0), np.max(vals, axis=0)
        #conf_int = np.array([np.min(vals, axis=0) * (2-self._confidence), np.max(vals, axis=0) * self._confidence]).T[0]

        if mn == mx == mean:
            conf_int = None
        else:
            conf_int = np.array([mn * (2-self._confidence), mx * self._confidence]).T[0]

        Plotly.plot_predictions(config=self.config, predictions=mean, conf_int=conf_int)


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

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_steps))

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_steps))

    def create_model(self):

        """
        Dobar za n_output = 1
        model.add(Dense(50, activation='relu', input_dim=self.config.n_steps))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))

        """
        model = Sequential()
        model.add(Dense(500, activation='relu', input_dim=self.config.n_steps))
        model.add(Dropout(0.2))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class LSTMModelWrapper(NNBaseModelWrapper):

    def create_model(self):

        model = Sequential()
        model.add(LSTM(300, activation='relu', return_sequences=True, input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(LSTM(300, activation='relu'))
        model.add(Dense(self.config.n_outputs))
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        """
        optimizer="nadam" ???? i za GRU
        """

        self.model = model


class BidirectionalLSTMModelWrapper(NNBaseModelWrapper):

    def create_model(self):
        """
        model = Sequential()
        model.add(Bidirectional(LSTM(500, activation='relu', return_sequences=True), input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(Bidirectional(LSTM(300, activation='relu')))
        model.add(Dense(1))
        """
        model = Sequential()
        model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True), input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(Bidirectional(LSTM(50, activation='relu')))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class TimeDistributedCNNLSTMModelWrapper(NNBaseModelWrapper):

    def create_model(self):
        """
        Isto ok - samo loši test
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=256, kernel_size=1, activation='relu'), input_shape=(None, self.config.n_steps, self.config.n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(300, activation='relu'))
        model.add(Dense(1))
        """
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=7, activation='relu'), input_shape=(None, self.config.n_steps, self.config.n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(300, activation='relu'))
        model.add(Dropout(0.2))
        # model.add(Dense(300, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(10, activation='relu'))
        model.add(Dense(self.config.n_outputs))
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_seq, self.config.n_steps, self.config.n_features))

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_seq, self.config.n_steps, self.config.n_features))


class CNNLSTMModelWrapper(NNBaseModelWrapper):

    def create_model(self):
        """
        # TODO tune kernel_size (=n_steps?)
        model = Sequential()
        model.add(ConvLSTM2D(return_sequences=True,
                             filters=256,
                             kernel_size=(1, 1),
                             activation='relu',
                             input_shape=(self.config.n_seq, 1, self.config.n_steps, self.config.n_features)))
        model.add(ConvLSTM2D(filters=256, kernel_size=(1, 1), return_sequences=True))
        model.add(ConvLSTM2D(filters=256, kernel_size=(1, 1), return_sequences=True))
        model.add(Flatten())
        model.add(Dense(1))
        """
        model = Sequential()
        model.add(ConvLSTM2D(return_sequences=True,
                             filters=64,
                             kernel_size=(1, 5),
                             activation='relu',
                             input_shape=(self.config.n_seq, 1, self.config.n_steps, self.config.n_features)))
        # model.add(ConvLSTM2D(filters=64, kernel_size=(1, 5), return_sequences=True))
        model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3), return_sequences=True, activation="relu"))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_seq, 1, self.config.n_steps, self.config.n_features))

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_seq, 1, self.config.n_steps, self.config.n_features))


class GRUModelWrapper(NNBaseModelWrapper):

    def create_model(self):

        model = Sequential()
        model.add(GRU(units=300, return_sequences=True, input_shape=(self.config.n_steps, self.config.n_features), activation='relu'))
        model.add(Dropout(0.1))
        #model.add(GRU(units=300, return_sequences=True, input_shape=(config.n_steps, config.n_features), activation='relu'))
        #model.add(Dropout(0.1))
        #model.add(GRU(units=300, return_sequences=True, input_shape=(config.n_steps, config.n_features), activation='relu'))
        #model.add(Dropout(0.1))
        model.add(GRU(300, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(self.config.n_outputs))
        # model.compile(optimizer=SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model


class CNNModelWrapper(NNBaseModelWrapper):

    def create_model(self):
        """
        filters=64, kernel=3, dense=10
        """
        """
        model.add(Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.config.n_outputs))
        """
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        # model.add(Dense(100, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.config.n_outputs))

        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_steps, self.config.n_features))

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_steps, self.config.n_features))
    """
    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0]*x.shape[1], x.shape[2]))

    def _reshape_predict_input(self, x):
        return x.reshape((1, len(x), 1))
    """


class MultichannelCNNModelWrapper(NNBaseModelWrapper):

    def create_model(self):

        model = Sequential()
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
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.config.n_steps, self.config.n_features)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.config.n_outputs))
        """
        model.compile(optimizer=self.optimizer, loss=self.loss_metric)

        self.model = model

    def _reshape_train_input(self, x):
        return x.reshape((x.shape[0], self.config.n_steps, self.config.n_features))

    def _reshape_predict_input(self, x):
        return x.reshape((1, self.config.n_steps, self.config.n_features))
