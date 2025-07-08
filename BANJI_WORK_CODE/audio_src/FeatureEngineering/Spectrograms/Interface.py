from abc import ABC, abstractmethod

import numpy as np
from scipy import signal


class FrequencySpectrumCalculator(ABC):

    @abstractmethod
    def make_spectrum(self, time_series: np.ndarray) -> np.ndarray:
        """
        Calculate the frequency spectrum of the given time series.

        Args:
            time_series (np.ndarray): The time series signal.

        Returns:
            np.ndarray: The frequency spectrum of the time series.
        """
        pass

    @abstractmethod
    def make_timeseries(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Reconstruct the time series from the given frequency spectrum.

        Args:
            spectrogram (np.ndarray): The frequency spectrum of the time series.

        Returns:
            np.ndarray: The reconstructed time series.
        """
        pass
