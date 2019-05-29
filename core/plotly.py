from core.config import Config
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class Plotly():

    # https://material.io/design/color/the-color-system.html#tools-for-picking-colors
    COLOR_PALETTE = [
        # All @500 colors from MD palette
        "#F44336", "#E91E63", "#9C27B0", "#673AB7", "#3F51B5", "#2196F3", "#03A9F4", "#00BCD4", "#009688", "#4CAF50",
        "#8BC34A", "#CDDC39", "#FFEB3B", "#FFC107", "#FF9800", "#FF5722", "#795548", "#9E9E9E",
        
        # All @900 colors from MD palette
        "#B71C1C", "#880E4F",  "#4A148C",  "#311B92",  "#1A237E",  "#0D47A1",  "#01579B",  "#006064",  "#004D40",
        "#1B5E20",  "#33691E",  "#827717",  "#F57F17",  "#FF6F00",  "#E65100",  "#BF360C",  "#3E2723",  "#212121",  "#263238"
    ]
    HISTORY_COLOR = "#2196F3" # blue
    REAL_COLOR = "#00E676" # green
    PREDICTED_COLOR = "#AA00FF" # "#D50000" # red
    ONE_STEP_PREDICTED_COLOR = "#AA00FF" # "#D50000" # red
    MULTI_STEP_PREDICTED_COLOR = "#FFC400" # "#F57C00" # orange
    MULTIPLE_PREDICTED_COLORS = ["#F44336", "#E91E63", "#5E35B1"] # ["#FFD600", "#F57C00", "#DD2C00"] # ["#0091EA", "#304FFE", "#6200EA"] # ["red", "orange", "purple"]
    CONFIDENCE_INTERVAL_COLOR = "#BDBDBD"
    ALPHA = 0.5
    ZOOM = -100 #None
    FIGURE_SIZE = (18, 10)

    _texts = list()

    @staticmethod
    def zoom(config: Config, predictions):

        if Plotly.ZOOM is not None:
            mn = config.test.index[0] + datetime.timedelta(Plotly.ZOOM)
            mx = config.test.index[-1] + datetime.timedelta(30)

            if len(predictions) > len(config.test):
                mx = config.test.index[0] + datetime.timedelta(len(predictions) + 30)

            plt.axis([mn, mx, 0, max([max(predictions)*1.1, max(config.all[mn:mx].y.values)*1.2])])


    @staticmethod
    def _show_texts():

        df = pd.DataFrame(Plotly._texts).sort_values("y")

        prev_y = None

        for index, row in df.iterrows():

            y = row["y"]

            if prev_y is not None and (prev_y > y or prev_y - y < 5):
                y = prev_y + 5

            plt.text(row["x"], y, row["text"], color=row["color"])
            prev_y = y

        Plotly._texts = list()

    @staticmethod
    def print_stats(real, predictions, info=None, return_stats=False):

        #if info is not None:
        #    print(info)

        mae = mean_absolute_error(real, predictions)
        mae = round(mae, 2)
        #print("MAE: {0:.2f}.".format(mae))

        rmse = math.sqrt(mean_squared_error(real, predictions))
        rmse = round(rmse, 2)
        #print("RMSE: {0:.2f}.".format(rmse))

        acc = (sum(predictions) / sum(real))
        if isinstance(acc, (list, tuple, np.ndarray)):
            acc = acc[0]

        acc = round(acc, 2)
        #print("ACC: {0:.2f}.".format(acc))

        print(("" if info is None else info + " ") + "MAE: {0:.2f}, RMSE: {1:.2f}, ACC: {2:.2f}".format(mae, rmse, acc))

        if return_stats:
            return mae, rmse, acc

    @staticmethod
    def _plot_score_text(df: pd.DataFrame, predictions, mae, acc, color):

        x = df.index[-1] + datetime.timedelta(days=5)
        y = predictions[-1][0] if type(predictions[-1]) is list else predictions[-1]
        text = "MAE: " + str(mae) + ", ACC: " + str(acc)

        Plotly._texts.append(dict({"x": x, "y": y, "text": text, "color": color}))

    @staticmethod
    def plot_predictions(config: Config, predictions, conf_int=None):

        train, test = config.train, config.test
        #mae, rmse, acc = Plotly.print_stats(test.y.values, predictions, return_stats=True)

        if config.verbose > 1 and len(predictions) > len(test):
            print("Predictions are larger than test, calculating loss on first {0} predictions.".format(len(test)))

        # Calculate error only on available test data, prediction can be larger than test set
        mae, rmse, acc = Plotly.print_stats(test.y.values, predictions[:len(test)], return_stats=True)

        plt.plot(train.y, label="y", color=Plotly.HISTORY_COLOR, alpha=Plotly.ALPHA)
        plt.plot(test.y, label="Real (test)", color=Plotly.REAL_COLOR, alpha=Plotly.ALPHA)

        #plt.plot(test.index, predictions, label="forecast", color=Plotly.PREDICTED_COLOR)
        predictions_index = [test.index[0] + datetime.timedelta(days=x) for x in range(len(predictions))]
        plt.plot(predictions_index, predictions, label="forecast", color=Plotly.PREDICTED_COLOR)

        Plotly._plot_score_text(test, predictions, mae, acc, Plotly.PREDICTED_COLOR)

        legend = list(["y (train)", "y (test)", "Forecast"])

        if conf_int is not None:
            plt.fill_between(predictions_index, conf_int[:, 0], conf_int[:, 1], color=Plotly.CONFIDENCE_INTERVAL_COLOR, alpha=Plotly.ALPHA, label="conf_int")
            legend.append("95% confidence interval")

        if config.verbose > 0 and len(predictions) > len(test):

            print("Adding possible values, average of previous years, after test set...")
            x = [config.all.index[-1] + datetime.timedelta(x) for x in range(0, len(predictions) - len(test))]

            y = config.all.groupby([config.all.index.month, config.all.index.day]).mean().y.values
            split_index = (x[0] - datetime.datetime(x[0].year, 1, 1)).days
            y_2 = []

            while len(y_2) < len(x):
                y_2 = np.concatenate([y_2, y[split_index:], y[:split_index]])

            y_2 = y_2[:len(x)]

            plt.plot(x, y_2, label="Possible values: AVG(previous years)", color=Plotly.CONFIDENCE_INTERVAL_COLOR)
            legend.append("Likely values (average of previous years)")

        Plotly._show_texts()
        Plotly.zoom(config, predictions)

        plt.legend(legend, loc="best")
        plt.rcParams["figure.figsize"] = Plotly.FIGURE_SIZE
        plt.show()

    @staticmethod
    def plot_train_predictions(config: Config, predictions, predictions_multistep=None, conf_int=None):

        train, test = config.train, config.test
        real = train.y.values[len(train) - len(predictions):len(train)]
        mae, rmse, acc = Plotly.print_stats(real, predictions, info="One-step Forecast:", return_stats=True)

        legend = list(["Real", "One-step Forecast"])

        plt.plot(train.y, label="Real", color=Plotly.REAL_COLOR, alpha=Plotly.ALPHA)
        plt.plot(train.index[len(train) - len(predictions):len(train)], predictions, label="One-step Forecast", color=Plotly.ONE_STEP_PREDICTED_COLOR)
        Plotly._plot_score_text(train, predictions, mae, acc, Plotly.ONE_STEP_PREDICTED_COLOR)

        if predictions_multistep is not None:
            legend.append("Recursive Multi-step Forecast")
            real = train.y.values[len(train) - len(predictions_multistep):len(train)]
            mae_ms, rmse_ms, acc_ms = Plotly.print_stats(real, predictions_multistep, info="Recursive Multi-step Forecast:", return_stats=True)

            plt.plot(train.index[len(train) - len(predictions_multistep):len(train)], predictions_multistep, label="Recursive Multi-step Forecast", color=Plotly.MULTI_STEP_PREDICTED_COLOR)
            Plotly._plot_score_text(train, predictions_multistep, mae_ms, acc_ms, Plotly.MULTI_STEP_PREDICTED_COLOR)

        if conf_int is not None:
            plt.fill_between(train.index[len(train) - len(predictions):len(train)], conf_int[:, 0], conf_int[:, 1], color=Plotly.CONFIDENCE_INTERVAL_COLOR, alpha=Plotly.ALPHA, label="conf_int")
            legend.append("95% confidence interval")

        Plotly._show_texts()
        plt.legend(legend, loc="best")
        plt.rcParams["figure.figsize"] = Plotly.FIGURE_SIZE
        plt.show()

    @staticmethod
    def plot_multiple_predictions(config: Config, mtpl_predictions, filter_gaps=None):

        all = config.all
        legend = list(["Real"])
        #colors = ["blue", "orange", "purple"]
        colors = Plotly.MULTIPLE_PREDICTED_COLORS

        plt.plot(all.y, label="Real", color=Plotly.HISTORY_COLOR, alpha=Plotly.ALPHA)

        longest_predictions = None

        for i, gap_pred in enumerate(mtpl_predictions):

            gap, predictions = gap_pred['gap'], gap_pred['predictions']

            if filter_gaps is not None and gap not in filter_gaps:
                continue

            if longest_predictions is None or len(predictions) > len(longest_predictions):
                longest_predictions = predictions

            test_l = all[-gap:]

            mae, rmse, acc = Plotly.print_stats(test_l.y.values, predictions, return_stats=True, info="Gap: " + str(gap))

            plt.plot(test_l.index, predictions, label="y" + str(gap), color=colors[i])
            Plotly._plot_score_text(test_l, predictions, mae, acc, colors[i])

            legend.append("Forecast (GAP: " + str(gap) + " )")

        Plotly._show_texts()
        Plotly.zoom(config, longest_predictions)
        plt.legend(legend, loc="best")
        plt.rcParams["figure.figsize"] = Plotly.FIGURE_SIZE
        plt.show()

    @staticmethod
    def plot_nn_train_history(history):
        print(history.history.keys())

        # summarize history for accuracy
        plt.plot(history.history["acc"])
        plt.plot(history.history["val_acc"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="best")
        plt.show()

        # summarize history for loss
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="brst")
        plt.show()

    @staticmethod
    def savefig(name, title=None):
        if title is not None:
            plt.title(title)
        plt.savefig(name)
        plt.close('all')

    @staticmethod
    def plot_seasonal_decompose(config):
        decomposed = sm.tsa.seasonal_decompose(config.all.y, freq=360)  # The frequncy is annual
        decomposed.plot()
        plt.rcParams["figure.figsize"] = Plotly.FIGURE_SIZE
        plt.show()

    @staticmethod
    def plot_autocorrelation(config):
        autocorrelation_plot(config.all.y)
        plt.rcParams["figure.figsize"] = Plotly.FIGURE_SIZE
        plt.show()

    @staticmethod
    def plot_acf(config):
        # Autocorrelation
        plot_acf(config.all.y, lags=365, title="Broj soba")
        # q = 14 - postoji statistički signifikantna autokorelacija sa 14 danom unatrag (363????)
        plt.rcParams["figure.figsize"] = Plotly.FIGURE_SIZE
        plt.show()

    @staticmethod
    def plot_pacf(config):
        # Partial Autocorrelation
        plot_pacf(config.all.y, lags=365, title="Broj soba")
        # p = 14
        plt.rcParams["figure.figsize"] = Plotly.FIGURE_SIZE
        plt.show()
