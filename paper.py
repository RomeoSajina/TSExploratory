"""
Za svaki hotel napraviti ts za 1.7., 1.8. i 1.9.

    Za svaki ts napraviti model prema odabranoj arhitekturi

        Spremit predviđanja za razdoblja 60, 30, 7


Rezultati:

hotel | model_info | target_date | train_time | n_parameters | p_train | p_train_ms | p_60 | p_30 | p_7

"""


from core.__init__ import *
from core.model.nn_model import NNBaseModelWrapper
from core.model.base_model import BaseModelWrapper
from sklearn.metrics import mean_absolute_error
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pickle
import seaborn as sns


ALL_MODELS = [
    PersistentModelWrapper,
    SeasonalPersistentModelWrapper,

    ARModelWrapper,
    MAModelWrapper,
    ARMAModelWrapper,
    ARIMAModelWrapper,
    SARIMAXModelWrapper,

    ProphetModelWrapper,

    [MLPModelWrapper, 8],
    [GRUModelWrapper, 11],
    [LSTMModelWrapper, 6],
    [CNNModelWrapper, 3],
    [MultiCNNModelWrapper, 3],
    [BidirectionalLSTMModelWrapper, 4],
    [AutoencoderMLPModelWrapper, 12],
    [AutoencoderCNNModelWrapper, 7],
    [AutoencoderMultiCNNModelWrapper, 3],
    [AutoencoderMLPGRUModelWrapper, 7],
    [AutoencoderMLPLSTMModelWrapper, 12]
]
#ALL_MODELS = [SARIMAXModelWrapper]


def plot_ts2(configs: list):

    legend = list()

    for index, element in enumerate(configs):

        hotel = element["hotel"]
        config = element["config"]

        plt.plot(config.all, label=hotel, color=Plotly.COLOR_PALETTE[::2][index])

        legend.append(hotel)

    plt.xlabel("Datum")
    plt.ylabel("Broj soba")
    plt.title("Rezervacije")
    plt.legend(legend, loc="best")
    Plotly.show()


def apply_for_models(hotel: str, config: Config, model_list: list, fnc, skip_fnc=None):

    for model_class in model_list:

        model_ver = 1

        if isinstance(model_class, list):
            model_ver = model_class[1]
            model_class = model_class[0]

        print(str(model_class), str(model_ver))

        config.apply_metadata(getattr(Metadata, "version_" + str(model_ver))())
        model = model_class(config)

        if isinstance(model, NNBaseModelWrapper):
            model.epochs = 1500

        model.save_load_model = False

        start = time.time()

        try:
            if skip_fnc is not None and skip_fnc(hotel, config, model):
                continue

            model.fit()
        except:
            model.model = None

        end = time.time()

        model.training_time = end - start # seconds

        fnc(hotel, config, model)


def save_stats(stats):
    #stats.to_csv(Config.base_dir() + "data/stats/model_predictions.csv", sep=";")
    with open(Config.base_dir() + "data/stats/model_predictions.pkl", 'wb') as output:
        pickle.dump(stats, output, pickle.HIGHEST_PROTOCOL)


def load_stats():

    def load_internal():
        #return pd.read_csv(Config.base_dir() + "data/stats/model_predictions.csv", sep=";", index_col=0)
        with open(Config.base_dir() + "data/stats/model_predictions.pkl", 'rb') as input:
            return pickle.load(input)

    try:
        return load_internal()
    except:

        columns = ["hotel", "model_info", "target_date", "train_time", "n_parameters", "p_train", "p_train_ms", "p_60", "p_30", "p_7"]
        #columns = ["hotel", "model_info", "for_date", "train_time", "n_parameters"]
        #for i, l in enumerate(["p_train", "p_train_ms", "p_60", "p_30", "p_7"]):
        #    columns.append(l + "_" + str(i))

        stats = pd.DataFrame(columns=columns)
        save_stats(stats)
        return stats


def load_extended_stats():

    stats = load_stats()

    assert len(stats.groupby(["hotel", "model_info", "target_date"]).size()) == len(stats), "There are duplicates of model results"

    """p_train | p_train_ms | p_60 | p_30 | p_7"""
    """
    for new_column in ["p_train_maes", "p_train_mae", "p_train_acc",
                       "p_train_ms_maes", "p_train_ms_mae", "p_train_ms_acc",
                       "p_60_maes", "p_60_mae", "p_60_acc",
                       "p_30_maes", "p_30_mae", "p_30_acc",
                       "p_7_maes", "p_7_mae", "p_7_acc"]:
        stats[new_column] = np.zeros(len(stats))
    """

    for target_date in [datetime.datetime(2018, 7, 1), datetime.datetime(2018, 8, 1), datetime.datetime(2018, 9, 1)]:

        configs = DataFactory.load_ts2(target_date=target_date, end_date=target_date - datetime.timedelta(days=60))

        for element in configs:
            hotel, config = element["hotel"], element["config"]

            filtered = (stats.hotel == hotel) & (stats.target_date == config.target_date)

            if not np.array(filtered).any():
                # Skip if no data found
                continue

            for num_c in [60, 30, 7]:

                c = "p_" + str(num_c)
                rc = "real_" + str(num_c)

                size = num_c

                stats.loc[filtered, rc] = stats[filtered][c].apply(lambda x: config.test.y.values[-size:])

                stats.loc[filtered, c + "_maes"] = \
                    stats[filtered][c].apply(lambda x: [round(mean_absolute_error([config.test.y.values[-size:][i]], [x[i]]), 2) for i in range(size)])

                stats.loc[filtered, c + "_mae"], stats.loc[filtered, c + "_acc"] = \
                    zip(*stats[filtered][c].apply(lambda x: np.array(StatsCollector.calculate_stats(config.test.y.values[-size:], x))[[0, 2]]))

            """
            Not working correctly!!!
            
            #tempy = stats[ [len(x[1]["p_train"])<20 for x in stats.iterrows()] ]#[["model_info", "p_train_ms"]]
            #tempy[["model_info", "hotel", "p_train"]]
            #stats = stats.drop(tempy.index)
            #len(stats)
            #save_stats(stats)
            
            for c in ["p_train", "p_train_ms"]:

                stats.loc[filtered, c + "_maes"] = \
                    stats[filtered][c].apply(lambda x: [round(mean_absolute_error([config.train.y.values[-len(x):][i]], [x[i]]), 2) for i in range(len(x))])

                stats.loc[filtered, c + "_mae"], stats.loc[filtered, c + "_acc"] = \
                    zip(*stats[filtered][c].apply(lambda x: np.array(StatsCollector.calculate_stats(config.train.y.values[-len(x):], x))[[0, 2]]))
            """
        stats.loc[:, "simple_model_name"] = stats.model_info.str.split("_").str[0]

    return stats


def collect_results(hotel: str, config: Config, model: BaseModelWrapper):

    if model.model is None:
        p_train, p_train_ms = np.zeros(1), np.zeros(1)
    else:
        p_train, p_train_ms = model.predict_on_train()

    p_train_ms = p_train_ms if p_train_ms is not None else np.zeros(1)

    mtpl_predictions = model.predict_multiple() if model.model is not None else [{"predictions": np.zeros(60)}, {"predictions": np.zeros(30)}, {"predictions": np.zeros(7)}]
    p_60 = mtpl_predictions[0]["predictions"]
    p_30 = mtpl_predictions[1]["predictions"]
    p_7 = mtpl_predictions[2]["predictions"]

    #hotel | model_info | for_date | train_time | n_parameters | p_train | p_train_ms | p_60 | p_30 | p_7

    new_values_list = [hotel, model.info(), config.target_date, model.training_time, model.n_params(),
                       np.array(p_train).flatten(), np.array(p_train_ms).flatten(),
                       np.array(p_60).flatten(), np.array(p_30).flatten(), np.array(p_7).flatten()]
    #new_values_list = [hotel, model.info(), config.target_date, model.training_time, model.n_params()]
    #for l in [p_train, p_train_ms, p_60, p_30, p_7]:
    #    new_values_list.extend(l)

    stats = load_stats()

    stats = stats.append(pd.DataFrame([new_values_list], columns=stats.columns.values), ignore_index=True)

    save_stats(stats)


def skip_collect_results(hotel: str, config: Config, model: BaseModelWrapper):

    stats = load_stats()

    already_done = np.array((stats.model_info == model.info()) & (stats.hotel == hotel) & (stats.target_date == config.target_date)).any()

    if not already_done:
        already_done = not reserve(hotel, config, model)

    return already_done


def reserve(hotel: str, config: Config, model: BaseModelWrapper):

    def load_rez():
        try:
            with open(Config.base_dir() + "data/stats/dibs.pkl", 'rb') as input:
                return pickle.load(input)
        except:
            columns = ["hotel", "model_info", "target_date"]
            rez = pd.DataFrame(columns=columns)
            save_rez(rez)
            return rez

    def save_rez(rez):
        with open(Config.base_dir() + "data/stats/dibs.pkl", 'wb') as output:
            pickle.dump(rez, output, pickle.HIGHEST_PROTOCOL)

    rez = load_rez()

    if len(rez[(rez.hotel == hotel) & (rez.model_info == model.info()) & (rez.target_date == config.target_date)]) > 0:
        return False

    rez.loc[len(rez)] = [hotel, model.info(), config.target_date]

    save_rez(rez)

    return True


def run():

    start = time.time()

    for target_date in [datetime.datetime(2018, 7, 1), datetime.datetime(2018, 8, 1), datetime.datetime(2018, 9, 1)]:

        configs = DataFactory.load_ts2(target_date=target_date, end_date=target_date - datetime.timedelta(days=60))

        for element in configs:
            hotel, config = element["hotel"], element["config"]
            print(hotel, config)

            apply_for_models(hotel, config, ALL_MODELS, collect_results, skip_fnc=skip_collect_results)
            import gc
            gc.collect()

    end = time.time()

    print(str(datetime.timedelta(seconds=end - start)))


####################### RUN #####################

stats = load_extended_stats()

#run()

stats = load_stats()
# stats[stats.model_info == "SARIMAX_p_1_d_0_q_1"].values

len(stats)



target_date = datetime.datetime(2018, 7, 1)
target_date = datetime.datetime(2018, 8, 1)
target_date = datetime.datetime(2018, 9, 1)
configs = DataFactory.load_ts2(target_date=target_date, end_date=target_date - datetime.timedelta(days=60))

plot_ts2(configs)













stats = load_extended_stats()
model_simple_names = stats.simple_model_name.unique()
model_in_focus = model_simple_names[0]
colors = {}

for i, td in enumerate(stats.target_date.unique()):
    colors[pd.to_datetime(str(td)).strftime("%d.%m.%Y.")] = Plotly.COLOR_PALETTE[::3][i]

subset = stats[(stats.simple_model_name == model_in_focus)]

assert len(subset.model_info.unique()) == 1, "Fail sa focus modela"

subset = subset.sort_values(["hotel", "hotel", "target_date"])


def plot_it(x):

    vals = x.p_60_maes

    clr = colors[x.target_date.strftime("%d.%m.%Y.")]

    sns.distplot(vals, hist=False, kde=True, kde_kws={'linewidth': 3}, color=clr, label=x.target_date.strftime("%d.%m.%Y."))


subset.apply(plot_it, axis=1)

plt.legend(colors, loc="best")
plt.title(model_in_focus)






def prepare_for_boxplot(subset, y_attr="p_60_maes"):

    bx_data = pd.DataFrame(columns=["hotel", "target_date", "y"])

    def f(x):
        ys = x[y_attr]

        for y in ys:
            bx_data.loc[len(bx_data)] = [x.hotel, x.target_date.strftime("%d.%m."), y]

    subset.apply(f, axis=1)

    return bx_data


stats = load_extended_stats()

model_in_focus = stats.simple_model_name.unique()[0]

for model_in_focus in stats.simple_model_name.unique():

    subset = stats[(stats.simple_model_name == model_in_focus)]

    bx_data = prepare_for_boxplot(subset)
    # https://seaborn.pydata.org/generated/seaborn.boxplot.html
    sns.boxplot(x="hotel", y="y", hue="target_date", data=bx_data, palette="Set3")
    #   sns.swarmplot(x="hotel", y="y", data=bx_data, color=".55")

    plt.title(model_in_focus) # + " (" + target_date.strftime("%d.%m.%Y.") + ")")
    Plotly.show()
    Plotly.savefigpkg("paper/" + model_in_focus + ".png")

    plt.close('all')



"""
n_hotels = 12
n_dates = 3

n_models = len(ALL_MODELS)

print("Total records: " + str(n_hotels*n_dates*n_models))
# 648


n_vrsta_kriticnih_modela = 3

n_sporih_modela = n_hotels*n_dates*n_vrsta_kriticnih_modela 

n_epocha = 1000

n_trajanje_epoche = 30 # s

s = n_sporih_modela * n_epocha * n_trajanje_epoche # s

h = n_sporih_modela * n_epocha * n_trajanje_epoche / 3600 # h

h / 24 # dana

"""






"""
def collect_results_on_models(hotel: str, config: Config, model_list: list):

    apply_for_models(hotel, config, model_list, collect_results)
"""


"""
target_date = datetime.datetime(2018, 7, 1)
target_date = datetime.datetime(2018, 8, 1)
target_date = datetime.datetime(2018, 9, 1)
configs = DataFactory.load_ts2(target_date=target_date, end_date=target_date - datetime.timedelta(days=60))

plot_ts2(configs)
"""



























"""
data = [1, 32, 2, 32, 32, 123]
plt.hist(data, bins=25, density=True, alpha=0.6, color='g')


hotels = [el["hotel"] for el in configs]

for h in hotels:

    subset = stats[(stats.hotel == h) & (stats.target_date == target_date)]

    def plot_it(x):

        vals = x.p_60
        label = target_date.strftime("%d_%m_%Y") + " - " + h + " - " + x.model_info

        sns.distplot(vals, hist=False, kde=True, kde_kws={'linewidth': 3}, label=label)

    subset.apply(plot_it, axis=1)


# Plot formatting
plt.legend(prop={'size': 16}, title='Errors')
#plt.title('Density Plot with Multiple Airlines')
#plt.xlabel('Delay (min)')
#plt.ylabel('Density')



target_date = datetime.datetime(2018, 7, 1)

model_colors = {}

for i, msn in enumerate([x.split("_")[0] for x in stats.model_info.unique()]):
    model_colors[msn] = Plotly.COLOR_PALETTE[::2][i]


subset = stats[(stats.target_date == target_date) & (stats.target_date == target_date)]

def plot_it(x):

    vals = x.p_60

    simple_name = x.model_info.split("_")[0]

    sns.distplot(vals, hist=False, kde=True, kde_kws={'linewidth': 3}, color=model_colors[simple_name]) # , label=x.model_info


stats.apply(plot_it, axis=1)

plt.legend(model_colors)
plt.title(target_date.strftime("%d_%m_%Y"))
"""




"""
config = DataFactory.load_ts()

acf_v, acf_ci = acf(config.all.y.values, nlags=365, alpha=.05)
lower_bound, upper_bound= acf_ci[:, 0]-acf_v, acf_ci[:, 1]-acf_v

tmp = [x > upper_bound[i] if x > 0 else x < lower_bound[i] for i, x in enumerate(acf_v)]

import numpy as np
np.where(np.array(tmp) == False)[0][0] - 1

plt.plot(acf_v)
plt.fill_between(range(len(acf_v)), upper_bound, lower_bound, color=Plotly.CONFIDENCE_INTERVAL_COLOR, alpha=Plotly.ALPHA, label="conf_int")



pacf_v, pacf_ci = pacf(config.all.y.values, nlags=365, alpha=.05)
lower_bound, upper_bound = pacf_ci[:, 0]-pacf_v, pacf_ci[:, 1]-pacf_v

tmp = [x > upper_bound[i] if x > 0 else x < lower_bound[i] for i, x in enumerate(pacf_v)]

import numpy as np
np.where(np.array(tmp) == False)[0][0] - 1

plt.plot(pacf_v)
plt.fill_between(range(len(pacf_v)), upper_bound, lower_bound, color=Plotly.CONFIDENCE_INTERVAL_COLOR, alpha=Plotly.ALPHA, label="conf_int")
"""