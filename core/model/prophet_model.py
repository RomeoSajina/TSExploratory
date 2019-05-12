from core.model.base_model import BaseModelWrapper
from fbprophet import Prophet
import pandas as pd
from core.config import Config


class ProphetModelWrapper(BaseModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self._model_fit_fnc = None

    def create_model(self):
        def model_fit_fnc(p_config: Config):
            new_model = Prophet(seasonality_prior_scale=5,
                                interval_width=0.95,
                                yearly_seasonality=True,
                                weekly_seasonality=True)

            # TODO: ovo ne bi smilo poboljšat model jer ljudi ki rezerviraju vecinon nisu iz HR
            new_model.add_country_holidays("HR")

            new_model.fit(p_config.train.reset_index().rename(columns={'X': 'ds', 'y': 'y'}))
            return new_model

        self._model_fit_fnc = model_fit_fnc

    def fit_model(self, refitting=False):
        self.model = self._model_fit_fnc(self.config)

    def predict(self, days=None):
        return self._predict(days)

    def _predict(self, days=None):

        if days is None:
            days = (self.config.target_date - self.config.end_date).days + 1

        test_ds = pd.concat([self.config.train, self.config.test])[-days:].reset_index().rename(columns={'X': 'ds', 'y': 'y'}).copy()

        # Remove y, just to be sure
        del test_ds['y']
        import datetime
        test_ds = pd.DataFrame([self.config.test.index[0] + datetime.timedelta(days=x) for x in range(0, days)], columns=["ds"])

        forecast = self.model.predict(df=test_ds)

        predictions = forecast.yhat.values

        return predictions

    def predict_on_train(self):

        test_ds = self.config.train.reset_index().rename(columns={'X': 'ds', 'y': 'y'}).copy()

        # Remove y, just to be sure
        del test_ds['y']

        forecast = self.model.predict(df=test_ds)

        predictions = forecast.yhat.values

        predictions_multistep = None
        return predictions, predictions_multistep


#plot_predictions(config, prophet_predict(config, model))
#plot_multiple_predictions(config, prophet_predict_multiple(config, model, model_fit_fnc=model_fit_fnc))

#plot_train_predictions(config, prophet_predict(config, model, days=len(config.all))[:-len(config.test)+1]) # Remove test set
