from core.model.base_model import StatsBaseModelWrapper
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARIMAResults, ARMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX
from core.config import Config
import numpy as np
import pandas as pd
import datetime


class ARIMAModelWrapper(StatsBaseModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self.p = 5
        self.d = 0
        self.q = 5
        self._model = None

    def adfuller_test(self):
        # Augmented Dickey-Fuller test
        adf = sm.tsa.stattools.adfuller(self.config.all.y)
        print("p-value y: {0:.10f}".format(float(adf[1])))
        # Null hipoteza je da TS nije stacionarna. Ako je p < 0.05 onda se odbacuje null hipoteza i prihvaća se hipoteza da je TS stacionarna
        # p-value y: 0.0013095158
        # Zaključak: nije potrebno diferenciranje
        return adf

    def create_model(self):

        self._model = ARIMA(self.config.train.y.values, order=(self.p, self.d, self.q))

    def fit_model(self, refitting=True):
        self.model = self._model.fit(trend='nc')
        #print(self.model.summary())

    def predict(self, days=None, return_confidence_interval=False):
        return self._predict(days, return_confidence_interval=return_confidence_interval)

    def _predict(self, days=None, return_confidence_interval=False):

        if days is None:
            days = (self.config.target_date - self.config.end_date).days + 1

        max_end = len(self.config.train) + len(self.config.test)
        conf_int = None

        try:
            predictions = self.model.predict(start=max_end - days, end=max_end - 1, dynamic=True) # Error if prediction with gap in train data (start>model.nobs)

            if return_confidence_interval:
                forecast, fcasterr, conf_int = self.model.forecast(steps=days, alpha=1.0 - self._confidence)
                predictions = forecast

        except:
            if self.config.verbose > 0:
                print("Exception when prediction: " + str(self.model))

            predictions = self.model.predict(start=self.model.nobs, end=max_end - 1, dynamic=True)

        if not isinstance(predictions, np.ndarray):
            predictions = predictions.values  # in case it is pandas Series

        predictions = predictions[-days:]

        # Can't be negative
        predictions[predictions < 0] = 0

        if return_confidence_interval:
            return predictions, conf_int

        return predictions

    def predict_on_train(self):

        predictions = self.model.fittedvalues  # ILI self.model.predict(start=0, end=len(config.train) - 1, dynamic=False)

        # lag = model_fit.model.order[0] # 14, model_fit.model.k_ma
        #lag = 14
        lag = self.p if self.p > self.q else self.q

        # dynamic=True => Recursive Multi-step forecast
        predictions_multistep = self.model.predict(start=lag, end=len(self.config.train) - 1, dynamic=True)
        predictions_multistep = np.concatenate([np.zeros(lag), predictions_multistep])

        # Actual vs Fitted
        # model_fit.plot_predict(start=lag, end=len(config.train) - 1, dynamic=True)
        # model_fit.plot_predict(dynamic=False)

        predictions[predictions < 0] = 0
        predictions_multistep[predictions_multistep < 0] = 0

        return predictions, predictions_multistep

    def load(self):
        #self.model = ARIMAResults.load(self.config.base_dir + "models/final/ARIMA.pkl")
        self.model = ARIMAResults.load(self._build_model_file_name())

    def save(self):
        #self.model.save(self.config.base_dir + "models/final/ARIMA.pkl")
        self.model.save(self._build_model_file_name())

    def n_params(self):
        return 0 if self.model is None else len(self.model.params)


class ARMAModelWrapper(ARIMAModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self.p = 5
        self.q = 5

    def create_model(self):
        self._model = ARMA(self.config.train.y.values, order=(self.p, self.q))

    def fit_model(self, refitting=True):
        self.model = self._model.fit(trend='nc')
        #print(self.model.summary())


class ARModelWrapper(ARIMAModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self.q = 0
        self.p = 9


class MAModelWrapper(ARIMAModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self.q = 25
        self.p = 0


class SARIMAXModelWrapper(ARIMAModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)
        self.p = 1
        self.q = 1
        self.m = 365

    def create_model(self):
        self._model = SARIMAX(self.config.train.y.values,
                              order=(self.p, self.d, self.q),
                              seasonal_order=(self.p, self.d, self.q, self.m),
                              trend='c',
                              enforce_stationarity=False,
                              enforce_invertibility=False)

    def fit_model(self, refitting=True):

        try:

            self.model = self._model.fit(maxiter=1000, disp=False)

        except Exception as e:

            if self.config.verbose > 0:
                print("Exception: " + str(e))
                self.extend_dataset()
                self.create_model()
                self.model = self._model.fit(maxiter=1000, disp=False)

            else:
                raise e

        #print(self.model.summary())

    def save(self):
        from sklearn.externals import joblib
        joblib.dump(self.model, self._build_model_file_name().replace(".pkl", ".sav"))

    def load(self):
        #self.model = SARIMAXResults.load(self._build_model_file_name())
        from sklearn.externals import joblib
        self.model = joblib.load(self._build_model_file_name().replace(".pkl", ".sav"))

    def extend_dataset(self):

        if self.config.verbose > 1:
            print("Extending dataset....")

        m = min(self.config.all.index)

        d = [m - datetime.timedelta(x) for x in reversed(range(1096))]

        a = self.config.all

        t = pd.DataFrame(dict({"X": d, "y": self.config.all.y.values}), columns=["X", "y"])
        t.set_index("X", inplace=True)

        a = t.append(a)

        #self.config.train = a[0:-60]
        #self.config.test = a[-60:]
        #self.config.all = pd.concat([self.config.train, self.config.test])
        self.config.set_train_and_test(a[0:-60], a[-60:])

        if self.config.verbose > 1:
            print("Extended dataset: " + self.config.all.index[0].strftime("%d_%m_%Y") + " - " + self.config.all.index[-1].strftime("%d_%m_%Y"))


class UnobservedComponentsModelWrapper(ARIMAModelWrapper):

    def __init__(self, config: Config):
        super().__init__(config)

    def create_model(self):
        self._model = sm.tsa.UnobservedComponents(self.config.train.y.values, 'local level')

    def fit_model(self, refitting=True):
        self.model = self._model.fit(maxiter=1000, disp=False)
        #print(self.model.summary())