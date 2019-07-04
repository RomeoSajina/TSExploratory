from core.model.base_model import StatsBaseModelWrapper
from fbprophet import Prophet
import pandas as pd
import datetime
import numpy as np
from core.config import Config


class ProphetModelWrapper(StatsBaseModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self._model_fit_fnc = None

    def create_model(self):
        def model_fit_fnc(p_config: Config):
            new_model = Prophet(seasonality_prior_scale=5,
                                interval_width=self._confidence,
                                yearly_seasonality=True,
                                weekly_seasonality=True)

            # TODO: ovo ne bi smilo poboljšat model jer ljudi ki rezerviraju vecinon nisu iz HR
            new_model.add_country_holidays("HR")

            new_model.fit(p_config.train.reset_index().rename(columns={'X': 'ds', 'y': 'y'}))
            return new_model

        self._model_fit_fnc = model_fit_fnc

    def fit_model(self, refitting=False):
        self.model = self._model_fit_fnc(self.config)

    def predict(self, days=None, return_confidence_interval=False):

        if return_confidence_interval:
            return self._predict(days)
        else:
            return self._predict(days)[0]

    def _predict(self, days=None):

        if days is None:
            days = (self.config.target_date - self.config.end_date).days + 1

        test_ds = pd.concat([self.config.train, self.config.test])[-days:].reset_index().rename(columns={'X': 'ds', 'y': 'y'}).copy()

        # Remove y, just to be sure
        del test_ds['y']

        test_ds = pd.DataFrame([self.config.test.index[0] + datetime.timedelta(days=x) for x in range(0, days)], columns=["ds"])

        forecast = self.model.predict(df=test_ds)

        predictions = forecast.yhat.values

        conf_int = np.array([forecast.yhat_lower, forecast.yhat_upper.values]).T

        return predictions, conf_int

    def predict_on_train(self):

        test_ds = self.config.train.reset_index().rename(columns={'X': 'ds', 'y': 'y'}).copy()

        # Remove y, just to be sure
        del test_ds['y']

        forecast = self.model.predict(df=test_ds)

        predictions = forecast.yhat.values

        predictions_multistep = None
        return predictions, predictions_multistep

    def n_params(self):
        return 0 if self.model is None else sum(len(np.array(x).flatten()) for x in [*self.model.params.values()])
