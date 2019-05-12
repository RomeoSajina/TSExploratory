from abc import ABC, abstractmethod
from core.config import Config
import pandas as pd
import datetime
import numpy as np
from core.plotly import Plotly


class BaseModelWrapper(ABC):

    #TODO: Box-Ljung test for residuals, if p-values are relatively large (cca 0.5), we can conclude that the residuals are not distinguishable from a white noise series.
    # Athanasopoulos str(75)

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
    def predict(self, days=None):
        raise NotImplementedError()

    @abstractmethod
    def predict_on_train(self):
        raise NotImplementedError()

    def predict_multiple(self):
        return self._predict_multiple()

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
            #else

            # Use the config if changed
            #self.config = copy_of_config
            preds = self.predict(gap)

            predictions.append(dict({"gap": gap, "predictions": preds}))

        # Reset config
        self.config = self.__config

        return predictions

    def plot_predict(self, days=None):
        predictions = self.predict(days=days)
        Plotly.plot_predictions(config=self.config, predictions=predictions)

    def plot_predict_multiple(self):
        predictions = self.predict_multiple()
        Plotly.plot_multiple_predictions(config=self.config, mtpl_predictions=predictions)

    def plot_train(self):
        predictions, predictions_multistep = self.predict_on_train()
        Plotly.plot_train_predictions(config=self.config, predictions=predictions, predictions_multistep=predictions_multistep)

    def plot_diagnostics(self):
        # https://machinelearningmastery.com/visualize-time-series-residual-forecast-errors-with-python/
        # TODO: napravit kao checkresiduals u R-u
        pass

    def _build_file_name(self):
        return self.__class__.__name__.replace("ModelWrapper", "")

    def _build_model_file_name(self):
        return self.config.base_dir + "models/final/" + self._build_file_name() + ".pkl"

    """
    def _construct_file_name(self):
        name = self.config.base_dir + "models/final/" + self.__class__.__name__.replace("ModelWrapper", "")
        return name

    def save_figure(self):
        name = self._construct_file_name().split(".")[0]
        print("Save fig: " + name + ".png")
        Plotly.savefig(name + ".png")
    """


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

    def predict(self, days=None):
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

