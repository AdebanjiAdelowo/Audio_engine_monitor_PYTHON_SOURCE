from abc import ABC, abstractmethod
from src.DataIO.Representation.Datum import Datum


class FeatureExtraction(ABC):

    """
    Abstract base class for all feature extraction classes.

    All feature extraction classes must implement the `extract_features` method.
    """

    @abstractmethod
    def extract_features(self, signal: Datum) -> Datum:
        """
        Extracts features from the given signal and returns a new Datum object
        containing the features.

        Args:
            signal: The input signal to extract features from.

        Returns:
            A new Datum object containing the extracted features.
        """
        pass
