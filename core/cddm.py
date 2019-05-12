from pandas import DataFrame
from abc import ABC, abstractmethod

# Concept drift detection method's (CDDM)


class ConceptDriftDetectionMethod(ABC):

    def __init__(self):
        self._train = DataFrame()
        self._new_values = DataFrame()
        self._predictions = list()
        super().__init__()

    def _init_attrs(self):
        self._train = DataFrame()
        self._new_values = DataFrame()
        self._predictions = list()

    def is_drifting(self, train: DataFrame, new_values: DataFrame, predictions: list):
        self._train = train
        self._new_values = new_values
        self._predictions = predictions

        is_drifting = self._is_drifting()

        # Release the resources
        self._init_attrs()

        return is_drifting

    @abstractmethod
    def _is_drifting(self):
        pass


class IgnoreConceptDriftDetectionMethod(ConceptDriftDetectionMethod):

    def _is_drifting(self):
        return False


class AlwaysConceptDriftDetectionMethod(ConceptDriftDetectionMethod):

    def _is_drifting(self):
        return True


class EarlyConceptDriftDetectionMethod(ConceptDriftDetectionMethod):

    def _is_drifting(self):
        # TODO: implement this method
        return False

