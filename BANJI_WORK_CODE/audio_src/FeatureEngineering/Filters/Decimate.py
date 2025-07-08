import numpy as np
import scipy

from src.FeatureEngineering.Filters.Interface import Filter


class AntiAliasingDecimation(Filter):

    def __init__(self, downsampling_factor: int = 3, order: int = 8) -> None:
        """
        Initializes the AntiAliasingDecimation filter.

        Parameters
        ----------
        downsampling_factor : int
            The downsampling factor. Defaults to 3.
        order : int
            The order of the filter. Defaults to 8.
        """
        self.__q = downsampling_factor
        self.__order = order

    def get_downsampling_factor(self) -> int:
        """
        Gets the downsampling factor of the AntiAliasingDecimation filter.

        Returns
        -------
        int
            The downsampling factor.
        """
        return self.__q

    def set_downsampling_factor(self, downsampling_factor: int):
        """
        Sets the downsampling factor of the AntiAliasingDecimation filter.

        Parameters
        ----------
        downsampling_factor : int
            The new downsampling factor.
        """
        self.__q = downsampling_factor

    def get_order(self) -> int:
        """
        Gets the order of the AntiAliasingDecimation filter.

        Returns
        -------
        int
            The order of the filter.
        """
        return self.__order

    def set_order(self, order: int):
        """
        Sets the order of the AntiAliasingDecimation filter.

        Parameters
        ----------
        order : int
            The new order of the filter.

        """
        self.__order = order

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies the AntiAliasingDecimation filter to the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The input signal to be filtered.
        sample_rate : int
            The sample rate of the input signal.

        Returns
        -------
        np.ndarray
            The filtered signal.
        """
        return scipy.signal.decimate(signal, self.__q, n=self.__order)


class Resample(Filter):

    def __init__(self, new_rate: int) -> None:
        """
        Initializes the Resample filter.

        Parameters
        ----------
        new_rate : int
            The new sample rate of the resampled signal.
        """
        self.__new_rate = new_rate

    def get_new_rate(self) -> int:
        """
        Gets the new sample rate of the resampled signal.

        Returns
        -------
        int
            The new sample rate of the resampled signal.
        """
        return self.__new_rate

    def set_new_rate(self, new_rate: int):
        """
        Sets the new sample rate of the resampled signal.

        Parameters
        ----------
        new_rate : int
            The new sample rate of the resampled signal.
        """
        self.__new_rate = new_rate

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies the Resample filter to the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The input signal to be filtered.
        sample_rate : int
            The sample rate of the input signal.

        Returns
        -------
        np.ndarray
            The resampled signal.
        """
        # Calculate the length of the resampled signal
        resampled_length = int(len(signal) * self.__new_rate / sample_rate)
        # Resample the signal
        return scipy.signal.resample(signal, resampled_length)
