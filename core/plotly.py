#from venusian.tests.fixtures import decorator

from core.config import Config
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from sklearn.exceptions import DataConversionWarning
from pandas.plotting import register_matplotlib_converters
import warnings
register_matplotlib_converters()
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
    MULTI_STEP_PREDICTED_COLOR = "#FFC107" # "#F57C00" # orange
    MULTIPLE_PREDICTED_COLORS = ["#9C27B0", "#F44336", "#FFC107", "#B71C1C", "#311B92", "#33691E"] # ["#F44336", "#9C27B0", "#3F51B5"] # ["#F44336", "#E91E63", "#5E35B1"]
    CONFIDENCE_INTERVAL_COLOR = "#BDBDBD"
    ALPHA = 0.5
    ZOOM = -305 #None
    FIGURE_SIZE = (18, 10)
    FONT_SIZE = 12.0

    plt.rcParams["figure.figsize"] = FIGURE_SIZE
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["date.autoformatter.year"] = "%Y"
    plt.rcParams["date.autoformatter.month"] = "%m.%Y"
    plt.rcParams["date.autoformatter.day"] = "%d.%m.%Y"
    plt.rcParams["date.autoformatter.hour"] = "%d.%m %H"
    plt.rcParams["date.autoformatter.minute"] = "%d %H:%M"
    plt.rcParams["date.autoformatter.second"] = "%H:%M:%S"
    plt.rcParams["date.autoformatter.microsecond"] = "%M:%S.%f"

    _texts = list()

    @staticmethod
    def set_figure_size(size):
        Plotly.FIGURE_SIZE = size
        plt.rcParams["figure.figsize"] = size

    @staticmethod
    def set_font_size(size):
        Plotly.FONT_SIZE = size
        plt.rcParams["font.size"] = float(size)

    @staticmethod
    def zoom(config: Config, predictions):

        if Plotly.ZOOM is not None:

            mn = config.test.index[0] + datetime.timedelta(Plotly.ZOOM) if len(config.test) > 0 else config.end_date +  datetime.timedelta(Plotly.ZOOM)
            mx = config.test.index[-1] + datetime.timedelta(30) if len(config.test) > 0 else config.train.index[-1] + datetime.timedelta(30)

            if 0 < len(config.test) < len(predictions):
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

        # FA (Forecasting Attainment)
        fa = (sum(predictions) / sum(real))
        if isinstance(fa, (list, tuple, np.ndarray)):
            fa = fa[0]

        fa = round(fa, 2)
        #print("fa: {0:.2f}.".format(fa))

        print(("" if info is None else info + " ") + "MAE: {0:.2f}, RMSE: {1:.2f}, FA: {2:.2f}".format(mae, rmse, fa))

        if return_stats:
            return mae, rmse, fa

    @staticmethod
    def _plot_score_text(df: pd.DataFrame, predictions, mae, fa, color):

        x = df.index[-1] + datetime.timedelta(days=5)
        y = predictions[-1][0] if type(predictions[-1]) is list else predictions[-1]
        text = "MAE: " + str(mae) + ", FA: " + str(fa)

        Plotly._texts.append(dict({"x": x, "y": y, "text": text, "color": color}))

    @staticmethod
    def plot_predictions(config: Config, predictions, conf_int=None):

        train, test = config.train, config.test
        #mae, rmse, fa = Plotly.print_stats(test.y.values, predictions, return_stats=True)

        if config.verbose > 1 and len(predictions) > len(test):
            print("Predictions are larger than test, calculating loss on first {0} predictions.".format(len(test)))

        # Calculate error only on available test data, prediction can be larger than test set
        mae, rmse, fa = Plotly.print_stats(test.y.values, predictions[:len(test)], return_stats=True)
        likely_mae, likely_rmse, likely_fa = 0., 0., 0.

        plt.plot(train.y, label="y", color=Plotly.HISTORY_COLOR, alpha=Plotly.ALPHA)
        plt.plot(test.y, label="Real (test)", color=Plotly.REAL_COLOR, alpha=Plotly.ALPHA)

        #plt.plot(test.index, predictions, label="forecast", color=Plotly.PREDICTED_COLOR)
        predictions_index = [test.index[0] + datetime.timedelta(days=x) for x in range(len(predictions))]
        plt.plot(predictions_index, predictions, label="forecast", color=Plotly.PREDICTED_COLOR)

        Plotly._plot_score_text(test, predictions, mae, fa, Plotly.PREDICTED_COLOR)

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

            likely_mae, likely_rmse, likely_fa = Plotly.print_stats(y_2, predictions[-len(y_2):], info="Likely values (average of previous years)", return_stats=True)

        Plotly._show_texts()
        Plotly.zoom(config, predictions)

        plt.legend(legend, loc="best")
        plt.xlabel("Datum")
        plt.ylabel("Broj soba")
        plt.title("Predviđanje broja rezervacija (" + test.index[0].strftime("%d.%m") + " - " + test.index[-1].strftime("%d.%m") + ")")
        #plt.draw()
        #return mae, rmse, fa, likely_mae, likely_rmse, likely_fa
        Plotly.show()

    @staticmethod
    def plot_train_predictions(config: Config, predictions, predictions_multistep=None, conf_int=None):

        train, test = config.train, config.test
        real = train.y.values[len(train) - len(predictions):len(train)]
        mae, rmse, fa = Plotly.print_stats(real, predictions, info="One-step Forecast:", return_stats=True)
        mae_ms, rmse_ms, fa_ms = 0., 0., 0.

        legend = list(["Real", "One-step Forecast"])

        plt.plot(train.y, label="Real", color=Plotly.REAL_COLOR, alpha=Plotly.ALPHA)
        plt.plot(train.index[len(train) - len(predictions):len(train)], predictions, label="One-step Forecast", color=Plotly.ONE_STEP_PREDICTED_COLOR)
        Plotly._plot_score_text(train, predictions, mae, fa, Plotly.ONE_STEP_PREDICTED_COLOR)

        if predictions_multistep is not None:
            legend.append("Recursive Multi-step Forecast")
            real = train.y.values[len(train) - len(predictions_multistep):len(train)]
            mae_ms, rmse_ms, fa_ms = Plotly.print_stats(real, predictions_multistep, info="Recursive Multi-step Forecast:", return_stats=True)

            plt.plot(train.index[len(train) - len(predictions_multistep):len(train)], predictions_multistep, label="Recursive Multi-step Forecast", color=Plotly.MULTI_STEP_PREDICTED_COLOR) #, alpha=Plotly.ALPHA
            Plotly._plot_score_text(train, predictions_multistep, mae_ms, fa_ms, Plotly.MULTI_STEP_PREDICTED_COLOR)

        if conf_int is not None:
            plt.fill_between(train.index[len(train) - len(predictions):len(train)], conf_int[:, 0], conf_int[:, 1], color=Plotly.CONFIDENCE_INTERVAL_COLOR, alpha=Plotly.ALPHA, label="conf_int")
            legend.append("95% confidence interval")

        Plotly._show_texts()
        plt.legend(legend, loc="best")
        plt.xlabel("Datum")
        plt.ylabel("Broj soba")
        plt.title("Predviđanje broja rezervacija nad trening podacima")
        #plt.draw()
        #return mae, rmse, fa,  mae_ms, rmse_ms, fa_ms
        Plotly.show()

    @staticmethod
    def plot_multiple_predictions(config: Config, mtpl_predictions, filter_gaps=None):

        all = config.all
        #legend = list(["Real"])
        legend = list(["y (train)", "y (test)"])
        #colors = ["blue", "orange", "purple"]
        colors = Plotly.MULTIPLE_PREDICTED_COLORS

        #plt.plot(all.y, label="Real", color=Plotly.HISTORY_COLOR, alpha=Plotly.ALPHA)

        plt.plot(config.train.y, label="y (train)", color=Plotly.HISTORY_COLOR, alpha=Plotly.ALPHA)
        plt.plot(config.test.y, label="Real (test)", color=Plotly.REAL_COLOR, alpha=Plotly.ALPHA)

        longest_predictions = None
        title_info = ""

        for i, gap_pred in enumerate(mtpl_predictions):

            gap, predictions = gap_pred['gap'], gap_pred['predictions']

            if filter_gaps is not None and gap not in filter_gaps:
                continue

            if longest_predictions is None or len(predictions) > len(longest_predictions):
                longest_predictions = predictions

            test_l = all[-gap:]

            mae, rmse, fa = Plotly.print_stats(test_l.y.values, predictions, return_stats=True, info="Gap: " + str(gap))

            plt.plot(test_l.index, predictions, label="y" + str(gap), color=colors[i])
            Plotly._plot_score_text(test_l, predictions, mae, fa, colors[i])

            #legend.append("Forecast (GAP: " + str(gap) + " )")
            legend.append("Forecast (" + test_l.index[0].strftime("%d.%m") + " - " + test_l.index[-1].strftime("%d.%m") + ")")
            #title_info += test_l.index[0].strftime("%d.%m") + " - " + test_l.index[-1].strftime("%d.%m") + ", "

        Plotly._show_texts()
        Plotly.zoom(config, longest_predictions)
        plt.legend(legend, loc="best")
        plt.xlabel("Datum")
        plt.ylabel("Broj soba")
        plt.title("Predviđanje broja rezervacija")# (" + title_info[:-2] + ")")

        # Fix for overlaping ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(len(ax.xaxis.get_major_ticks()) - len(mtpl_predictions) - 1))

        Plotly.show()

    @staticmethod
    def plot_multiple_predictions2(config: Config, mtpl_predictions):

        total_mae, total_fa = list(), list()

        idx = np.array(np.where(config.all.index == config.end_date)).flatten()[0]

        train, test = config.all[:idx], config.all[idx:]

        legend = list(["y (train)", "y (test)"])

        colors = Plotly.MULTIPLE_PREDICTED_COLORS

        # plt.plot(all.y, label="Real", color=Plotly.HISTORY_COLOR, alpha=Plotly.ALPHA)

        plt.plot(train.y, label="y (train)", color=Plotly.HISTORY_COLOR, alpha=Plotly.ALPHA)
        plt.plot(test.y, label="Real (test)", color=Plotly.REAL_COLOR, alpha=Plotly.ALPHA)

        longest_predictions = None
        title_info = ""

        for i, gap_pred in enumerate(mtpl_predictions):

            gap, predictions = gap_pred['gap'], gap_pred['predictions']

            if longest_predictions is None or len(predictions) > len(longest_predictions):
                longest_predictions = predictions

            test_l = test[:gap]

            mae, rmse, fa = Plotly.print_stats(test_l.y.values, predictions, return_stats=True, info="Gap: " + str(gap))

            plt.plot(test_l.index, predictions, label="y" + str(gap), color=colors[i])
            Plotly._plot_score_text(test_l, predictions, mae, fa, colors[i])

            #legend.append("Forecast (GAP: " + str(gap) + " )")
            legend.append("Forecast (" + test_l.index[0].strftime("%d.%m") + " - " + test_l.index[-1].strftime("%d.%m") + ")")
            #title_info += test_l.index[0].strftime("%d.%m") + " - " + test_l.index[-1].strftime("%d.%m") + ", "

            margin = len(train) + gap
            train, test = config.all[:margin], config.all[margin:]

            total_fa.append(fa)
            total_mae.append(mae)

        Plotly._show_texts()
        Plotly.zoom(config, longest_predictions)
        plt.legend(legend, loc="best")
        plt.xlabel("Datum")
        plt.ylabel("Broj soba")
        plt.title("Predviđanje broja rezervacija (MAE: " + str(round(np.mean(total_mae), 2)) + ", FA: " + str(round(np.mean(total_fa), 2)) + ")") # (" + title_info[:-2] + ")")
        Plotly.show()

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
        Plotly.show()

    @staticmethod
    def savefig(name, title=None):
        if title is not None:
            plt.title(title)
        plt.savefig(name)
        plt.close('all')

    @staticmethod
    def savefigpkg(name, title=None):
        Plotly.savefig(Config.base_dir() + "/figures/" + name, title)

    @staticmethod
    def plot_multiple_and_train(config: Config, predictions, predictions_multistep, train_title, mtpl_predictions, mtpl_title):

        size = Plotly.FIGURE_SIZE

        Plotly.set_figure_size((18, 6.5))

        plt.subplot(121)
        Plotly.plot_train_predictions(config, predictions=predictions, predictions_multistep=predictions_multistep)
        plt.title(train_title)
        plt.gca().figure.autofmt_xdate()

        plt.subplot(122)
        Plotly.plot_multiple_predictions(config, mtpl_predictions)
        plt.title(mtpl_title)
        plt.gca().figure.autofmt_xdate()

        #plt.subplots_adjust(top=.95, bottom=0.08, left=0.05, right=0.95, hspace=0.25, wspace=0.2)
        plt.subplots_adjust(top=.95, bottom=0.08, left=0.05, right=0.91, hspace=0.25, wspace=0.25)

        Plotly.set_figure_size(size)

    @staticmethod
    def plot_seasonal_decompose(config, filter=None):
        decomposed = sm.tsa.seasonal_decompose(config.all.y, freq=360)  # The frequncy is annual

        size = Plotly.FIGURE_SIZE

        if filter is not None:

            if filter == "trend":
                Plotly.set_figure_size((18, 4))
                decomposed.trend.plot()
                plt.title("Trend rezervacija")
            #TODO: add other components

        else:
            decomposed.plot()

        plt.xlabel("Datum")
        Plotly.show()
        Plotly.set_figure_size(size)

    @staticmethod
    def plot_autocorrelation(config):
        autocorrelation_plot(config.all.y)
        Plotly.show()

    @staticmethod
    def plot_acf(config):
        # Autocorrelation
        plot_acf(config.all.y, lags=365, title="ACF")
        Plotly.show()

    @staticmethod
    def plot_pacf(config):
        # Partial Autocorrelation
        plot_pacf(config.all.y, lags=365, title="PACF")
        Plotly.show()

    @staticmethod
    def plot_model_stats(stats, metric="test_mae", sequentially=True):

        legend = list()
        for i, g in enumerate(stats[stats.sequentially == sequentially].groupby(["model"])):

            model_name = g[0]
            model_stats = g[1].sort_values("version")

            plt.plot(model_stats.version.values, model_stats[metric].values, color=Plotly.COLOR_PALETTE[::2][i])

            #plt.text(max(model_stats.version.values)+1, model_stats[metric].values[-1], model_name, color=Plotly.COLOR_PALETTE[i])
            legend.append(model_name)

        plt.title("Model statistics (sequentially: {0}, metric: {1})".format(str(sequentially).lower(), metric))
        plt.xlabel("Version")
        plt.ylabel("Metric value")
        plt.legend(legend, loc="best")
        Plotly.show()

    @staticmethod
    def plot_ts(config: Config, frame_seasons=False):

        plt.plot(config.all, label="ts", color=Plotly.HISTORY_COLOR)

        if frame_seasons:

            my = min(config.all.y)
            mx = max(config.all.y)

            start = min(config.all.index)

            while start < config.all.index[-1]:

                diff = datetime.timedelta( (start.replace(year=start.year+1) - start).days )

                rect = Rectangle((start, my), diff, mx, linewidth=2, edgecolor=Plotly.COLOR_PALETTE[15], facecolor='none') #, alpha=Plotly.ALPHA

                plt.gca().add_patch(rect)

                start = start + diff

        plt.xlabel("Datum")
        plt.ylabel("Broj soba")
        plt.title("Rezervacije")
        Plotly.show()

    @staticmethod
    def show():
        plt.rcParams["figure.figsize"] = Plotly.FIGURE_SIZE
        plt.tight_layout()
        plt.show()
