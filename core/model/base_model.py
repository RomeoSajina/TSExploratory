from abc import ABC, abstractmethod
from core.config import Config
import pandas as pd
import datetime
import numpy as np
from core.plotly import Plotly


class BaseModelWrapper(ABC):

    def __init__(self, config: Config):
        super().__init__()
        # Private config
        self.__config = config
        # Public config
        self.config = config
        self.model = None
        self.save_load_model = True
        self._confidence = .95

    def save(self):
        pass

    def load(self):
        raise NotImplementedError()

    @abstractmethod
    def create_model(self):
        pass

    def fit(self):

        self.create_model()

        if self.save_load_model:
            try:
                self.load()
            except:
                self.fit_model()
                self.save()
                if self.config.verbose > 0:
                    print("Saved model: " + self.__class__.__name__ + ", " + self.model.__class__.__name__)
        else:
            self.fit_model()

    @abstractmethod
    def fit_model(self, refitting=False):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, days=None, return_confidence_interval=False):
        raise NotImplementedError()

    @abstractmethod
    def predict_on_train(self):
        raise NotImplementedError()

    def predict_multiple(self):
        return self._predict_multiple()

    def predict_multiple2(self):

        """
        PSEUDO:

            foreach gap:

                if there exist a previous gap:

                    new_values = train[-prev_gap:]

                    # Prepare datasets to be like previous
                    train, test = shrinked set without new values, extended set with new values

                    old_data_predictions = predict the same range as new values

                    if concept is drifting:
                        retrain the model

                predictions = predict the next range

                train, test = extended set with new values, shrinked test set without new values

        """

        predictions = list()

        prev_gap = None

        for gap in self.config.gaps:

            train, test = self.config.train_and_test

            if prev_gap is not None:

                new_values = train[-prev_gap:]

                if self.config.verbose > 0:
                    print("New values available after: " + str(new_values.index[0].strftime("%d.%m.%Y")) + ", size: " + str(len(new_values)))

                train_prev, test_prev = train[:-prev_gap], pd.concat([train[-prev_gap:], test])

                self.config.set_train_and_test(train_prev, test_prev)

                old_data_predictions = self.predict(prev_gap)

                if self.config.verbose > 1:
                    print("Predictions of the range of new values with start date: " + str(new_values.index[0].strftime("%d.%m.%Y")) +
                          ", size: " + str(len(old_data_predictions)) + ", gap: " + str(prev_gap))

                if self.config.concept_is_drifting(new_values, old_data_predictions):
                    # Reset train and test data
                    self.config.set_train_and_test(train, test)

                    #if self.config.verbose > 1:
                    #    print("Initiating retraining, last sync date: " + self.config.min_date.strftime("%d.%m.%Y"))

                    self.fit_model(refitting=True)

                # Set new margin where system agreed to be up-to-date
                self.config.min_date = self.config.train.index[-1]

                # Reset train and test data
                self.config.set_train_and_test(train, test)

            # Predict new range
            preds = self.predict(gap)

            predictions.append(dict({"gap": gap, "predictions": preds}))

            # Dividie by new gap and set new history and future (train, test)
            margin = len(train) + gap
            train_l, test_l = self.config.all[:margin], self.config.all[margin:]
            self.config.set_train_and_test(train_l, test_l)

            # Notify new predictions
            self.config.notify_new_predictions(preds)

            # Set gap for next round
            prev_gap = gap

        return predictions

    def _predict_multiple(self):

        train, test = self.__config.train_and_test

        predictions = list()

        #copy_of_config = self.__config.copy()
        self.config = self.__config.copy()

        for gap in self.__config.gaps:

            end_date_l = test.index[-1] - datetime.timedelta(days=gap)

            if end_date_l > train.index[-1]:
                # New data is available that aren't included i train set (new available data is fresher than last in train set)

                all = pd.concat([train, test])

                # Dividie by new end date
                train_l, test_l = all[:end_date_l], all[end_date_l:]
                self.config.set_train_and_test(train_l, test_l)

                new_values = test[train.index[-1]:end_date_l]

                if self.config.verbose > 0:
                    print("New values available after end_date: " + str(end_date_l.strftime("%d.%m.%Y")) + ", size: " + str(len(new_values)))

                # Try to predict with old model. Some DDM's will use predictions to calculate error
                #self.config = copy_of_config
                pds_with_old_model = self.predict(gap)

                # copy_of_config still has the old values
                #if copy_of_config.concept_is_drifting(new_values, pds_with_old_model):
                if self.config.concept_is_drifting(new_values, pds_with_old_model):
                    # Only if concept has drifted take new values and fit the refit the model
                    #copy_of_config.set_train_and_test(train_l, test_l)
                    #self.config.set_train_and_test(train_l, test_l)
                    #self.config = copy_of_config
                    self.fit_model(refitting=True)

            # Use the config if changed
            #self.config = copy_of_config
            preds = self.predict(gap)

            predictions.append(dict({"gap": gap, "predictions": preds}))

        # Reset config
        self.config = self.__config

        return predictions

    def plot_predict(self, days=None, show_confidence_interval=True):

        if show_confidence_interval:
            predictions, conf_int = self.predict(days=days, return_confidence_interval=True)
        else:
            predictions, conf_int = self.predict(days=days, return_confidence_interval=False), None

        Plotly.plot_predictions(config=self.config, predictions=predictions, conf_int=conf_int)

    def plot_predict_multiple(self):
        predictions = self.predict_multiple()
        Plotly.plot_multiple_predictions(config=self.config, mtpl_predictions=predictions)

    def plot_predict_multiple2(self):
        predictions = self.predict_multiple2()
        Plotly.plot_multiple_predictions2(config=self.config, mtpl_predictions=predictions)

    def plot_train(self):
        predictions, predictions_multistep = self.predict_on_train()
        Plotly.plot_train_predictions(config=self.config, predictions=predictions, predictions_multistep=predictions_multistep)

    def plot_multiple_and_train(self):
        predictions, predictions_multistep = self.predict_on_train()
        Plotly.plot_multiple_and_train(config=self.config,
                                       predictions=predictions, predictions_multistep=predictions_multistep, train_title=self._build_file_name() + "_train",
                                       mtpl_predictions=self.predict_multiple(), mtpl_title=self._build_file_name() + "_prediction")

    def plot_diagnostics(self):
        # https://machinelearningmastery.com/visualize-time-series-residual-forecast-errors-with-python/
        # TODO: napravit kao checkresiduals u R-u
        pass

    def n_params(self):
        return 0

    def info(self):
        return self.__class__.__name__.replace("ModelWrapper", "")

    def _build_file_name(self):
        return self.__class__.__name__.replace("ModelWrapper", "")

    def _build_model_file_name(self):
        return self.config.base_dir + "models/final/" + self._build_file_name() + ".pkl"

    def save_train_figure(self):
        self.__save_figure("train")

    def save_prediction_figure(self):
        self.__save_figure("prediction")

    def save_figure(self, info="", device="svg"):
        self.__save_figure(info, device)

    def __save_figure(self, info, device="png"):
        model_info = self.__class__.__name__.replace("ModelWrapper", "") + "_" + info
        name = self.config.base_dir + "figures/models/" + model_info
        Plotly.savefig(name + "." + device, model_info)
    """
    def _construct_file_name(self):
        name = self.config.base_dir + "models/final/" + self.__class__.__name__.replace("ModelWrapper", "")
        return name

    def save_figure(self):
        name = self._construct_file_name().split(".")[0]
        print("Save fig: " + name + ".png")
        Plotly.savefig(name + ".png")
    """


class StatsBaseModelWrapper(BaseModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)


# Naive forecast
class PersistentModelWrapper(BaseModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self.save_load_model = False

    def create_model(self):
        pass

    def load(self):
        pass

    def fit_model(self, refitting=False):
        pass

    def predict(self, days=None, return_confidence_interval=False):

        if return_confidence_interval:
            return self._predict(days), None #TODO: implement confidence interval
        else:
            return self._predict(days)

    def _predict(self, days=None):

        if days is None:
            days = (self.config.target_date - self.config.end_date).days + 1

        predictions = np.repeat(self.config.train.y.values[-1], days)

        return predictions

    def predict_on_train(self):
        #predictions = np.concatenate([np.array([0]), self.config.train.y.shift(1).values[1:]])
        #predictions_multistep = np.repeat(self.config.train.y.values[0], len(self.config.train))
        predictions = self.config.train.y.shift(1).values[1:]
        predictions_multistep = np.repeat(self.config.train.y.values[0], len(self.config.train))
        return predictions, predictions_multistep


# Seasonal naive forecast
class SeasonalPersistentModelWrapper(PersistentModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self.season = 365

    def _predict(self, days=None):

        if days is None:
            days = (self.config.target_date - self.config.end_date).days + 1

        #predictions = self.config.train.y.values[-self.season:-self.season+days]
        predictions = self.config.train.y.values[-self.season:][:days]

        return predictions

    def predict_on_train(self):

        season, train = self.season, self.config.train
        years = int(len(train)/season)
        #lag = np.repeat(0, season)

        #predictions = np.concatenate([lag, train.y.shift(season).values[season:]])
        predictions = train.y.shift(season).values[season:]

        first_seaason = train.values[:season]

        mts = np.tile(first_seaason.reshape(first_seaason.shape[0]), years + 1)[:len(train) - season]

        #predictions_multistep = np.concatenate([lag, mts])
        predictions_multistep = mts

        # Note: first year of predictions overlaps with first year of predictions_multistep
        # In case that predictions are ploted, first year of firstly ploted predictions will be hidden

        return predictions, predictions_multistep

