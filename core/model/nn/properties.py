import configparser
import collections
from core.config import Config


class Properties:

    FILE_NAME = "models/final/model.properties"
    LOSS_KEY = "loss"
    LAST_EPOCH_KEY = "last_epoch"

    @staticmethod
    def _build_section_name(model): #: NNBaseModelWrapper
        return model.__class__.__name__.replace("ModelWrapper", "") + "_seq_" + str(model.train_sequentially) + "_ver_" + str(model.version)

    @staticmethod
    def _file_name():
        return Config.base_dir() + Properties.FILE_NAME

    @staticmethod
    def load(model): #: NNBaseModelWrapper

        #model_properties = configparser.ConfigParser()
        model_properties = configparser.ConfigParser({}, collections.OrderedDict)
        model_properties.read(Properties._file_name())

        section = Properties._build_section_name(model)

        if model_properties.has_section(section):

            mp = model_properties[section]

            if model.config.verbose > 1:
                print("Loading properties for {0}:\n{1}".format(section, model_properties.items(section=section)))

            model.model_loss = float(mp[Properties.LOSS_KEY])
            model.last_epoch = int(mp[Properties.LAST_EPOCH_KEY])

    @staticmethod
    def save(model): #: NNBaseModelWrapper

        #model_properties = configparser.ConfigParser()
        model_properties = configparser.ConfigParser({}, collections.OrderedDict)
        model_properties.read(Properties._file_name())

        section = Properties._build_section_name(model)

        if not model_properties.has_section(section):
            model_properties.add_section(section)

        mp = model_properties[section]
        mp[Properties.LOSS_KEY] = str(model.model_loss)
        mp[Properties.LAST_EPOCH_KEY] = str(model.last_epoch)

        model_properties._sections = collections.OrderedDict(sorted(model_properties._sections.items(), key=lambda t: (t[0].split("_ver_")[0], int(t[0].split("_ver_")[1]))))

        if model.config.verbose > 1:
            print("Saving properties for {0}:\n{1}".format(section, model_properties.items(section=section)))

        with open(Properties._file_name(), "w") as fp:
            model_properties.write(fp)

