import math
import pandas as pd
import numpy as np
from core.model.base_model import BaseModelWrapper
from core import Config
from sklearn.metrics import mean_squared_error, mean_absolute_error


class StatsCollector:

    @staticmethod
    def calculate_stats(real, predictions):

        if predictions is None:
            return None, None, None

        mae = mean_absolute_error(real, predictions)
        mae = round(mae, 2)

        rmse = math.sqrt(mean_squared_error(real, predictions))
        rmse = round(rmse, 2)

        # FA (Forecasting Attainment)
        fa = sum(np.array(predictions).flatten()) / sum(np.array(real).flatten())
        if isinstance(fa, (list, tuple, np.ndarray)):
            fa = fa[0]

        fa = round(fa, 2)

        return mae, rmse, fa

    @staticmethod
    def _folder():
        return Config.base_dir() + "data/stats/"

    @staticmethod
    def load_stats(name="stats"):
        return pd.read_csv(StatsCollector._folder() + name + ".csv", index_col=0)

    @staticmethod
    def save_stats(stats, name="stats", index=True):
        stats.to_csv(StatsCollector._folder() + name + ".csv", index=index)

    @staticmethod
    def reset_mtpl_stats():
        stats = pd.DataFrame(columns=["code", "model", "version", "sequentially",
                                      "train_rmse", "train_mae", "train_fa",
                                      "train_ms_rmse", "train_ms_mae", "train_ms_fa",
                                      "test_rmse_60", "test_mae_60", "test_fa_60",
                                      "test_rmse_30", "test_mae_30", "test_fa_30",
                                      "test_rmse_7", "test_mae_7", "test_fa_7",
                                      "n_parameters"])
        StatsCollector.save_mtlp_stats(stats)

    @staticmethod
    def load_mtlp_stats():
         try:
             return StatsCollector.load_stats("mtpl_stats")
         except:
                StatsCollector.reset_mtpl_stats()
         return StatsCollector.load_stats("mtpl_stats")

    @staticmethod
    def save_mtlp_stats(stats):
        StatsCollector.save_stats(stats, "mtpl_stats")

    @staticmethod
    def collect_mtpl_model_stats(config: Config, model: BaseModelWrapper):

        train, test = config.train_and_test

        predictions, predictions_multistep = model.predict_on_train()

        train_mae, train_rmse, train_fa = StatsCollector.calculate_stats(config.train.y.values[len(train) - len(predictions):], predictions)

        if predictions_multistep is None:
            train_ms_mae, train_ms_rmse, train_ms_fa = None, None, None
        else:
            train_ms_mae, train_ms_rmse, train_ms_fa = StatsCollector.calculate_stats(config.train.y.values[len(train) - len(predictions_multistep):], predictions_multistep)

        mtpl_predictions = model.predict_multiple()
        test_mae_60, test_rmse_60, test_fa_60 = StatsCollector.calculate_stats(config.all[-60:].y.values, mtpl_predictions[0]["predictions"])
        test_mae_30, test_rmse_30, test_fa_30 = StatsCollector.calculate_stats(config.all[-30:].y.values, mtpl_predictions[1]["predictions"])
        test_mae_7, test_rmse_7, test_fa_7 = StatsCollector.calculate_stats(config.all[-7:].y.values, mtpl_predictions[2]["predictions"])

        """
        for i, gap_pred in enumerate(mtpl_predictions):

            gap, preds = gap_pred['gap'], gap_pred['predictions']

            test_l = config.all[-gap:]

            mae, rmse, fa = StatsCollector.calculate_stats(test_l.y.values, preds)
        """

        stats = StatsCollector.load_mtlp_stats()

        new_values_list = [model.info(), model.__class__.__name__.replace("ModelWrapper", ""),
                           config.version, True, #model.train_sequentially
                           train_rmse, train_mae, train_fa,
                           train_ms_rmse, train_ms_mae, train_ms_fa,
                           test_rmse_60, test_mae_60, test_fa_60,
                           test_rmse_30, test_mae_30, test_fa_30,
                           test_rmse_7, test_mae_7, test_fa_7,
                           model.n_params()]

        #stats.loc[len(stats)] = new_values_list
        stats = stats.append(pd.DataFrame([new_values_list], columns=stats.columns.values), ignore_index=True)

        StatsCollector.save_mtlp_stats(stats)

    @staticmethod
    def split_mtpl_stats():

        import sys
        import importlib
        importlib.import_module('core')

        def get_model_class(model_str):
            return getattr(sys.modules["core"], model_str + "ModelWrapper")

        from core.model.nn_model import NNBaseModelWrapper
        from core.model.base_model import StatsBaseModelWrapper

        stats = StatsCollector.load_mtlp_stats()

        """
        stats = stats[(stats.model.str.contains("RandomDropout") == False) & (stats.model.str.contains("ResNet") == False) & (stats.model.str.contains("TimeDistributed") == False) & (stats.model != "CNNLSTM") ]
        """

        stats_models = stats[[issubclass(get_model_class(x), StatsBaseModelWrapper) for x in stats.model]]
        nn_models = stats[[issubclass(get_model_class(x), NNBaseModelWrapper) for x in stats.model]]
        persistent_models = stats[(~stats.isin(stats_models)) & (~stats.isin(nn_models))].dropna()

        #ignore_cols = ["code", "version", "sequentially", "train_rmse", "train_ms_rmse", "test_rmse_60", "test_rmse_30", "test_rmse_7"]
        ignore_cols = ["code", "version", "sequentially", "train_rmse", "train_ms_rmse", "test_rmse_60", "test_rmse_30",
                       "test_rmse_7", "train_rmse", "train_mae", "train_fa", "train_ms_rmse", "train_ms_mae", "train_ms_fa",]
        nn_ignore_cols = list(ignore_cols)
        nn_ignore_cols.remove("version")

        stats_models = stats_models.loc[:, [x not in ignore_cols for x in stats_models.columns]]
        nn_models = nn_models.loc[:, [x not in nn_ignore_cols for x in nn_models.columns]]
        persistent_models = persistent_models.loc[:, [x not in ignore_cols for x in persistent_models.columns]]

        """
        c_names = ["Model", "Verzija",
                  "MAE(train One-Step)", "FA(train One-Step)",
                  "MAE(train Multi-Step)", "FA(train Multi-Step)",
                  "MAE(test Multi-Step) 60 dana", "FA(test Multi-Step) 60 dana",
                  "MAE(test Multi-Step) 30 dana", "FA(test Multi-Step) 30 dana",
                  "MAE(test Multi-Step) 7 dana",  "FA(test Multi-Step) 7 dana",
                  "Broj parametara"]
        """
        c_names = ["Model", "Verzija",
                   "MAE(60)", "FA(60)",
                   "MAE(30)", "FA(30)",
                   "MAE(7)", "FA(7)",
                   "Broj parametara"]

        c_names2 = list(c_names)
        c_names2.remove("Verzija")

        nn_models.columns = c_names
        stats_models.columns = c_names2
        persistent_models.columns = c_names2

        StatsCollector.save_stats(nn_models, "out/nn_models", False)
        StatsCollector.save_stats(stats_models, "out/stats_models", False)
        StatsCollector.save_stats(persistent_models, "out/persistent_models", False)

        """
        for gr in enumerate(nn_models.groupby(["Model"])):
            name = gr[1][0]
            items = gr[1][1]
            StatsCollector.save_stats(items, "out/"+name, False)
        """

    @staticmethod
    def load_mtpl_stats_with_avg():

        stats = StatsCollector.load_mtlp_stats()
        stats.loc[:, "test_avg_mae"] = sum([stats.test_mae_60, stats.test_mae_30, stats.test_mae_7]) / 3.
        stats.loc[:, "test_avg_fa"] = sum([stats.test_fa_60, stats.test_fa_30, stats.test_fa_7]) / 3.
        stats.test_avg_mae = stats.test_avg_mae.round(2)
        stats.test_avg_fa = stats.test_avg_fa.round(2)

        # ERR = MAE + (MAE * | 1 - FA |) -> Test faza
        stats.loc[:, "test_error"] = stats.test_avg_mae + (stats.test_avg_mae * abs(1 - stats.test_avg_fa))
        stats.test_error = stats.test_error.round(2)


        """
        stats = stats[stats.model.str.contains("RandomDropout") == False]
        
        stats.sort_values("test_avg_mae").loc[:, ["code", "test_avg_mae"]]
        
        stats[:][["model", "test_avg_mae", "test_avg_fa"]]

        
        config.verbose=0
        for model_class in ALL:

            model_ver = 1
        
            if isinstance(model_class, list):
                model_ver = model_class[1]
                model_class = model_class[0]
        
            config.apply_metadata(getattr(Metadata, "version_" + str(model_ver))())
            model = model_class(config)
        
            print(stats[stats.code == model.info()][:][["model", "version", "test_avg_mae", "test_avg_fa"]].values)

        
        print("version".ljust(10, ' '), "model".ljust(30, ' '), "test_avg_mae".ljust(20, ' '), "test_avg_fa".ljust(20, ' '), "test_error".ljust(20, ' '))
        for index, row in stats.iterrows():
            #print("".ljust(70, "-"))
            print(str(row["version"]).ljust(10, ' '), row["model"].ljust(30, ' '), str(row["test_avg_mae"]).ljust(20, ' '), str(row["test_avg_fa"]).ljust(20, ' '), str(row["test_error"]).ljust(20, ' '))

        
        filtered = stats[stats['model'].isin(["MLP", "GRU", "LSTM", "CNN", "MultiCNN", "BidirectionalLSTM", "AutoencoderMLP", "AutoencoderMultiCNN"])]
        Plotly.plot_model_stats(filtered, "test_avg_mae")
        """

        return stats