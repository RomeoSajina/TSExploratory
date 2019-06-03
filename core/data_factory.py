import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
import random
from core import Config
from core import Plotly


class DataFactory:

    @staticmethod
    def _prebuild_data():

        statuses = ["EF", "EO", "EP", "F", "NA", "NF", "O", "P", "PF", "PO"]
        channels = ["A", "I", "B"]

        data = pd.read_csv(Config.base_dir() + "data/reservations_all.csv",
                           parse_dates=["DATUM_KREIRANJA", "DATUM_OD", "DATUM_DO", "DATUM_STORNA", "VRIJEME_ZAMRZAVANJA"],
                           low_memory=False)

        g = data.groupby(["HOTEL", "GODINA", "SIF_REZERVACIJE"])
        mx = g["VRIJEME_ZAMRZAVANJA"].transform(max)

        data = data[data["VRIJEME_ZAMRZAVANJA"] == mx]

        data.STATUS_REZERVACIJE.fillna("F")
        data = data[[x in statuses for x in data.STATUS_REZERVACIJE]]
        data = data[[x in channels for x in data.KANAL_ID]]

        columns = ["DATUM_KREIRANJA", "DATUM_OD", "DATUM_DO", "BROJ_SOBA_BOOK"]

        data = data[columns]

        data.to_csv("data/reservations.csv")

    @staticmethod
    def load_ts(end_date=None, target_date=None):

        def load_data():
            return pd.read_csv(Config.base_dir() + "data/reservations.csv",
                               parse_dates=["DATUM_KREIRANJA", "DATUM_OD", "DATUM_DO"],
                               low_memory=False)
        try:
            data = load_data()
        except FileNotFoundError:
            DataFactory._prebuild_data()
            data = load_data()

        data.DATUM_OD = data.DATUM_OD.dt.normalize()
        data.DATUM_DO = data.DATUM_DO.dt.normalize()
        data.DATUM_KREIRANJA = data.DATUM_KREIRANJA.dt.normalize()

        if target_date is None:
            target_date = datetime.datetime(2018, 7, 1)

        if end_date is None:
            end_date = datetime.datetime(2018, 5, 3)

        min_date = datetime.datetime(2015, 7, 1)
        max_date = target_date

        data["DATUM_OD_C"] = pd.to_datetime([datetime.date(2018, 2, 28) if x == datetime.date(2016, 2, 29) else x.replace(year=2018) for x in data.DATUM_OD.dt.date.values])
        data["DATUM_DO_C"] = pd.to_datetime([datetime.date(2018, 2, 28) if x == datetime.date(2016, 2, 29) else x.replace(year=2018) for x in data.DATUM_DO.dt.date.values])

        target_date_data = data[(data.DATUM_OD_C <= target_date) & (data.DATUM_DO_C > target_date)]

        range_dates = [min_date + datetime.timedelta(days=x) for x in range(0, (max_date - min_date).days + 1)]

        assert min_date == range_dates[0] and max_date == range_dates[len(range_dates) - 1], "Range of dates generated incorrectly"


        elem_list = []

        for index in range(len(range_dates)):

            obs_date = range_dates[index]

            date_data = target_date_data[target_date_data.DATUM_KREIRANJA == obs_date]

            # Ignore 29.2. and add it to 28.2.
            if obs_date.date() == datetime.date(2016, 2, 29):
                prev = list(filter(lambda s: s["X"].date() == datetime.date(2016, 2, 28), elem_list))[0]
                prev["y"] += date_data.BROJ_SOBA_BOOK.sum()
                continue

            ts_elem = dict()
            ts_elem["X"] = obs_date
            ts_elem["y"] = date_data.BROJ_SOBA_BOOK.sum()

            elem_list.append(ts_elem)

        ts = pd.DataFrame(elem_list)

        ts = ts.set_index("X", drop=True)

        #return ts, min_date, end_date, target_date
        return Config.build(ts, min_date, end_date, target_date)

    @staticmethod
    def load_ts_in_range(start_date, end_date, diff_between):

        configs = list()

        for d in [start_date + datetime.timedelta(days=x) for x in range(0, (end_date - start_date).days + 1)]:
            configs.append(DataFactory.load_ts(target_date=d, end_date=d-datetime.timedelta(diff_between)))

        return configs

    @staticmethod
    def load_ts_in_range_flattern(start_date=None, end_date=None, diff_between=60):

        if start_date is None:
            start_date = datetime.datetime(2018, 6, 15)

        if end_date is None:
            end_date = datetime.datetime(2018, 8, 15)

        file_name = "data/flattern_ts_in_range-" + start_date.strftime("%d_%m_%Y") + "-" + end_date.strftime("%d_%m_%Y") + ".npy"

        try:
            return np.load(Config.base_dir() + file_name) # "data/flattern_ts_in_range.npy"
        except FileNotFoundError:
            pass

        configs = DataFactory.load_ts_in_range(start_date=start_date, end_date=end_date, diff_between=diff_between)

        train = []

        for c in configs:
            train = np.concatenate([train, c.train.y.values])

        np.save(Config.base_dir() + file_name, train) #"data/flattern_ts_in_range.npy"

        return train

    @staticmethod
    def decompose_by_year(data, target_date):

        decomposed = list()

        for year in range(min(data.index.year), max(data.index.year) + 1):

            y = data[(data.index > target_date.replace(year=year)) & (data.index <= target_date.replace(year=year + 1))].y.values

            if len(y) > 0:
                decomposed.append(dict({'year': year+1, 'x': list(range(-len(y), 0)), 'y':y}))

        return np.array(decomposed)

    @staticmethod
    def plot_yearly(target_date=datetime.datetime(2018, 7, 1)):

        config = DataFactory.load_ts(target_date=target_date)

        colors = Plotly.COLOR_PALETTE[::3] #["orange", "green", "purple"]
        legend = list()

        for i, dcmp in enumerate(DataFactory.decompose_by_year(config.all, target_date)):

            year, x, y = dcmp['year'], dcmp['x'], dcmp['y']

            plt.plot(x, y, color=colors[i])

            legend.append(str(year))

        plt.legend(legend, loc="best")
        plt.xlabel("Broj dana")
        plt.ylabel("Broj soba")
        plt.title("Kreiranje rezervacija soba za " + target_date.strftime("%d.%m."))

    @staticmethod
    def plot_for_target_dates(target_dates=[datetime.datetime(2018, 7, 1)], x_is_distance=True):

        #colors = ["red", "pink", "orange", "yellow", "green", "blue", "purple", "black"]
        #colors = random.choices([k[0] for k in matplotlib.colors.cnames.items()], k=len(target_dates))
        colors = Plotly.COLOR_PALETTE

        legend = list()

        for i, td in enumerate(target_dates):

            config = DataFactory.load_ts(target_date=td)

            dcmps = DataFactory.decompose_by_year(config.all, td)

            df = pd.DataFrame.from_records(dcmps, columns=["year", "x", "y"], index="year")

            # Ako prva godina ima manje podataka nego ostale godine
            # napuni nedostatak sa podacima iz sljedece godine
            len_0 = len(df.y[df.index[0]])
            len_1 = len(df.y[df.index[1]])
            if len_0 < len_1:
                df.y[df.index[0]] = np.concatenate([df.y[df.index[1]][0:len_1-len_0], df.y[df.index[0]]])
                df.x[df.index[0]] = df.x[df.index[1]]

            y_mean = np.mean(df.y, axis=0)

            if x_is_distance:
                x = df.x[df.index[0]]
            else:
                x = [td - datetime.timedelta(days=x) for x in reversed(range(0, len(y_mean)))]

            plt.plot(x, y_mean, color=colors[i])

            legend.append(td.strftime("%d.%m"))

        plt.legend(legend, loc="best")
        plt.xlabel("Broj dana")
        plt.ylabel("Broj soba")
        plt.title("Prosječna kreiranja rezervacija soba za datume (godine: 2016-2018)")
        plt.rcParams["figure.figsize"] = Plotly.FIGURE_SIZE
        plt.show()

    @staticmethod
    def build_aggregated():

        def build_extended():
            data = pd.read_csv(Config.base_dir() + "data/reservations_all.csv",
                               parse_dates=["DATUM_KREIRANJA", "DATUM_OD", "DATUM_DO", "DATUM_STORNA", "VRIJEME_ZAMRZAVANJA"],
                               low_memory=False)

            g = data.groupby(["HOTEL", "GODINA", "SIF_REZERVACIJE"])
            mx = g["VRIJEME_ZAMRZAVANJA"].transform(max)

            data = data[data["VRIJEME_ZAMRZAVANJA"] == mx]

            columns = ["DATUM_KREIRANJA", "KANAL_ID", "DATUM_OD", "DATUM_DO", "BROJ_OSOBA_BOOK",
                       "BROJ_SOBA_BOOK", "STORNO", "DATUM_STORNA", "VRIJEME_ZAMRZAVANJA", "HOTEL"]

            data = data[columns]

            data.to_csv(Config.base_dir() + "data/reservations_extended.csv")
            return data

        def load_extended():
            data = pd.read_csv(Config.base_dir() + "data/reservations_extended.csv",
                               parse_dates=["DATUM_KREIRANJA", "DATUM_OD", "DATUM_DO", "DATUM_STORNA", "VRIJEME_ZAMRZAVANJA"],
                               index_col=0,
                               low_memory=False)
            return data

        try:
            data = load_extended()
        except:
            data = build_extended()

        data.DATUM_OD = data.DATUM_OD.dt.normalize()
        data.DATUM_DO = data.DATUM_DO.dt.normalize()
        data.DATUM_KREIRANJA = data.DATUM_KREIRANJA.dt.normalize()

        data["DATUM_OD_C"] = pd.to_datetime([datetime.date(2018, 2, 28) if x == datetime.date(2016, 2, 29) else x.replace(year=2018) for x in data.DATUM_OD.dt.date.values])
        data["DATUM_DO_C"] = pd.to_datetime([datetime.date(2018, 2, 28) if x == datetime.date(2016, 2, 29) else x.replace(year=2018) for x in data.DATUM_DO.dt.date.values])

        min_date = datetime.datetime(2018, 6, 15)
        max_date = datetime.datetime(2018, 9, 15)

        range_dates = [min_date + datetime.timedelta(days=x) for x in range(0, (max_date - min_date).days + 1)]

        assert min_date == range_dates[0] and max_date == range_dates[-1], "Range of dates generated incorrectly"

        elem_list = []

        for index in range(len(range_dates)):

            obs_date = range_dates[index]

            obs_date_data = data[(data.DATUM_OD_C <= obs_date) & (data.DATUM_DO_C > obs_date)]

            g = obs_date_data.groupby(["HOTEL", "KANAL_ID", "DATUM_KREIRANJA"])

            for group in g.groups:
                hotel = group[0]
                kanal = group[1]
                datum = group[2]

                filtered_data = obs_date_data[(obs_date_data.HOTEL == hotel) & (obs_date_data.KANAL_ID == kanal) & (obs_date_data.DATUM_KREIRANJA == datum)]

                ts_elem = dict()
                ts_elem["DATUM_KREIRANJA"] = datum
                ts_elem["HOTEL"] = hotel
                ts_elem["KANAL"] = kanal
                ts_elem["ZA_DATUM"] = obs_date
                ts_elem["BROJ_SOBA_BOOK"] = filtered_data.BROJ_SOBA_BOOK.sum()

                elem_list.append(ts_elem)

        df = pd.DataFrame(elem_list)
        df.to_csv(Config.base_dir() + "data/reservations_aggregated.csv", sep=";")

    @staticmethod
    def load_stats():
        return pd.read_csv(Config.base_dir() + "data/stats.csv", index_col=0)

    @staticmethod
    def save_stats(stats):
        stats.to_csv(Config.base_dir() + "data/stats.csv")

    """
    DataFactory.plot_yearly()
    
    target_dates = [datetime.datetime(2018, 6, 1),
                    datetime.datetime(2018, 7, 1),
                    datetime.datetime(2018, 8, 1),
                    datetime.datetime(2018, 9, 1)]
    
    DataFactory.plot_for_target_dates(target_dates=target_dates, x_is_distance=True)
    DataFactory.plot_for_target_dates(target_dates=target_dates, x_is_distance=False)
                                                    
    # Cijeli 7 mj
    target_dates = [datetime.datetime(2018, 7, 1) + datetime.timedelta(x) for x in range(0, 31)]
    DataFactory.plot_for_target_dates(target_dates=target_dates, x_is_distance=False)
    
    """