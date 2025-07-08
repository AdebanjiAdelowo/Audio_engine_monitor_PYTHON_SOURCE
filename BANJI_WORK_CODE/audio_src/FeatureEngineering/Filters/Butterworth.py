from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt

from src.FeatureEngineering.Filters.Interface import Filter


class LowPassFilter(Filter):

    def __init__(self, cutoff: float, order: int, analog: bool) -> None:
        """
        Construct a LowPassFilter.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency of the filter.
        order : int
            The order of the filter.
        analog : bool
            Whether to use an analog filter, or a digital filter.

        Returns
        -------
        None
        """
        self.__cutoff = cutoff
        self.__order = order
        self.__analog = analog

    def get_cutoff(self):
        """
        Gets the cutoff frequency of the filter.

        Returns
        -------
        float
            The cutoff frequency of the filter.
        """
        return self.__cutoff

    def set_cutoff(self, cutoff: float):
        """
        Sets the cutoff frequency of the filter.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency of the filter.

        Returns
        -------
        None
        """
        self.__cutoff = cutoff

    def get_order(self):
        """
        Gets the order of the filter.

        Returns
        -------
        int
            The order of the filter.
        """
        return self.__order

    def set_order(self, order: int):
        """
        Sets the order of the filter.

        Parameters
        ----------
        order : int
            The order of the filter.

        Returns
        -------
        None
        """
        self.__order = order

    def get_analog(self) -> bool:
        """
        Gets the analog filter property of the filter.

        Returns
        -------
        bool
            True if the filter is an analog filter, False if it is a digital filter.
        """
        return self.__analog

    def set_analog(self, analog: bool):
        """
        Sets the analog filter property of the filter.

        Parameters
        ----------
        analog : bool
            True if the filter is an analog filter, False if it is a digital filter.

        Returns
        -------
        None
        """
        self.__analog = analog

    def apply_filter(self, signal: np.ndarray, sample_rate) -> np.ndarray:
        """
        Applies the filter to the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The signal to which the filter is to be applied.
        sample_rate : int
            The sample rate of the signal.

        Returns
        -------
        np.ndarray
            The filtered signal.
        """
        return self.__butter_lowpass_filter(signal, sample_rate)

    def __butter_lowpass(self, cutoff: float, sample_rate: int, order: int, analog: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the coefficients for a Butterworth low-pass filter.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency of the filter.
        sample_rate : int
            The sample rate of the signal.
        order : int
            The order of the filter.
        analog : bool
            Whether the filter is an analog filter, or a digital filter.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The coefficients for the filter.
        """
        nyquist = sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=analog, output='ba')
        return b, a

    def __butter_lowpass_filter(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies a Butterworth low-pass filter to the given data.

        Parameters
        ----------
        data : np.ndarray
            The data to which the filter is to be applied.
        sample_rate : int
            The sample rate of the data.

        Returns
        -------
        np.ndarray
            The filtered data.
        """
        # Calculate the coefficients for the filter
        b, a = self.__butter_lowpass(self.__cutoff, sample_rate, self.__order, self.__analog)
        # Apply the filter
        y = filtfilt(b, a, data)
        return y


class HighPassFilter(Filter):

    def __init__(self, cutoff: float, order: int, analog: bool) -> None:
        """
        Construct a HighPassFilter.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency of the filter.
        order : int
            The order of the filter.
        analog : bool
            Whether the filter is an analog filter, or a digital filter.
        """
        self.__cutoff = cutoff
        self.__order = order
        self.__analog = analog

    def get_cutoff(self) -> float:
        """
        Gets the cutoff frequency of the filter.

        Returns
        -------
        float
            The cutoff frequency of the filter.
        """
        return self.__cutoff

    def set_cutoff(self, cutoff: float):
        """
        Sets the cutoff frequency of the filter.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency of the filter.
        """
        self.__cutoff = cutoff

    def get_order(self) -> int:
        """
        Gets the order of the filter.

        Returns
        -------
        int
            The order of the filter.
        """
        return self.__order

    def set_order(self, order: int):
        """
        Sets the order of the filter.

        Parameters
        ----------
        order : int
            The order of the filter.
        """
        self.__order = order

    def get_analog(self) -> bool:
        """
        Gets whether the filter is an analog filter or a digital filter.

        Returns
        -------
        bool
            True if the filter is an analog filter, False if it is a digital filter.
        """
        return self.__analog

    def set_analog(self, analog: bool):
        """
        Sets whether the filter is an analog filter or a digital filter.

        Parameters
        ----------
        analog : bool
            True if the filter is an analog filter, False if it is a digital filter.
        """
        self.__analog = analog

    def apply_filter(self, signal: np.ndarray, sample_rate) -> np.ndarray:
        """
        Applies the filter to the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The signal to which the filter is to be applied.
        sample_rate : int
            The sample rate of the signal.

        Returns
        -------
        np.ndarray
            The filtered signal.
        """
        return self.__butter_highpass_filter(signal, sample_rate)

    def __butter_highpass(self, cutoff: float, sample_rate: int, order: int, analog: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the coefficients for a Butterworth high-pass filter.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency of the filter.
        sample_rate : int
            The sample rate of the signal.
        order : int
            The order of the filter.
        analog : bool
            Whether the filter is an analog filter, or a digital filter.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The coefficients for the filter.
        """
        nyquist = sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=analog, output='ba')
        return b, a

    def __butter_highpass_filter(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies a Butterworth high-pass filter to the given data.

        Parameters
        ----------
        data : np.ndarray
            The data to which the filter is to be applied.
        sample_rate : int
            The sample rate of the data.

        Returns
        -------
        np.ndarray
            The filtered data.
        """
        # Calculate the coefficients for the filter
        b, a = self.__butter_highpass(self.__cutoff, sample_rate, self.__order, self.__analog)
        # Apply the filter
        y = filtfilt(b, a, data)
        return y


class BandPassFilter(Filter):

    def __init__(self, lowcut: float, highcut: float, order: int, analog: bool) -> None:
        """
        Construct a BandPassFilter.

        Parameters
        ----------
        lowcut : float
            The lower cutoff frequency of the filter.
        highcut : float
            The upper cutoff frequency of the filter.
        order : int
            The order of the filter.
        analog : bool
            Whether the filter is an analog filter, or a digital filter.
        """
        self.__lowcut = lowcut
        self.__highcut = highcut
        self.__order = order
        self.__analog = analog

    def get_lowcut(self) -> float:
        """
        Gets the lower cutoff frequency of the filter.

        Returns
        -------
        float
            The lower cutoff frequency of the filter.
        """
        return self.__lowcut

    def set_lowcut(self, lowcut: float):
        """
        Sets the lower cutoff frequency of the filter.

        Parameters
        ----------
        lowcut : float
            The lower cutoff frequency of the filter.
        """
        self.__lowcut = lowcut

    def get_highcut(self) -> float:
        """
        Gets the upper cutoff frequency of the filter.

        Returns
        -------
        float
            The upper cutoff frequency of the filter.
        """
        return self.__highcut

    def set_highcut(self, highcut: float):
        """
        Sets the upper cutoff frequency of the filter.

        Parameters
        ----------
        highcut : float
            The upper cutoff frequency of the filter.
        """
        self.__highcut = highcut

    def get_order(self) -> int:
        """
        Gets the order of the filter.

        Returns
        -------
        int
            The order of the filter.
        """
        return self.__order

    def set_order(self, order: int):
        """
        Sets the order of the filter.

        Parameters
        ----------
        order : int
            The order of the filter.
        """
        self.__order = order

    def get_analog(self) -> bool:
        """
        Gets the analog filter property of the filter.

        Returns
        -------
        bool
            True if the filter is an analog filter, False if it is a digital filter.
        """
        return self.__analog

    def set_analog(self, analog: bool):
        """
        Sets the analog filter property of the filter.

        Parameters
        ----------
        analog : bool
            True if the filter is an analog filter, False if it is a digital filter.
        """
        self.__analog = analog

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies the filter to the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The signal to which the filter is to be applied.
        sample_rate : int
            The sample rate of the signal.

        Returns
        -------
        np.ndarray
            The filtered signal.
        """
        # Apply the filter
        return self.__butter_bandpass_filter(signal, sample_rate)

    def __butter_bandpass(self, lowcut: float, highcut: float, sample_rate: int, order: int, analog: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the coefficients for a Butterworth band-pass filter.

        Parameters
        ----------
        lowcut : float
            The lower cutoff frequency of the filter.
        highcut : float
            The upper cutoff frequency of the filter.
        sample_rate : int
            The sample rate of the signal.
        order : int
            The order of the filter.
        analog : bool
            Whether the filter is an analog filter, or a digital filter.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The coefficients for the filter.
        """
        # Calculate nyquist frequency
        nyquist = sample_rate / 2
        # Calculate normalized frequencies
        low = lowcut / nyquist
        high = highcut / nyquist
        # Calculate the coefficients for the filter
        b, a = butter(order, [low, high], btype='band', analog=analog, output='ba')
        return b, a

    def __butter_bandpass_filter(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies the Butterworth band-pass filter to the given data.

        Parameters
        ----------
        data : np.ndarray
            The data to which the filter is to be applied.
        sample_rate : int
            The sample rate of the data.

        Returns
        -------
        np.ndarray
            The filtered data.
        """
        # Calculate the coefficients for the filter
        b, a = self.__butter_bandpass(self.__lowcut, self.__highcut, sample_rate, self.__order, self.__analog)
        # Apply the filter
        y = filtfilt(b, a, data)
        return y


class BandStopFilter(Filter):

    def __init__(self, lowcut: float, highcut: float, order: int, analog: bool) -> None:
        """
        Construct a BandStopFilter.

        Parameters
        ----------
        lowcut : float
            The lower cutoff frequency of the filter.
        highcut : float
            The upper cutoff frequency of the filter.
        order : int
            The order of the filter.
        analog : bool
            Whether the filter is an analog filter, or a digital filter.
        """
        self.__lowcut = lowcut
        self.__highcut = highcut
        self.__order = order
        self.__analog = analog

    def get_lowcut(self) -> float:
        """
        Gets the lower cutoff frequency of the filter.

        Returns
        -------
        float
            The lower cutoff frequency of the filter.
        """
        return self.__lowcut

    def set_lowcut(self, lowcut: float):
        """
        Sets the lower cutoff frequency of the filter.

        Parameters
        ----------
        lowcut : float
            The lower cutoff frequency of the filter.
        """
        self.__lowcut = lowcut

    def get_highcut(self) -> float:
        """
        Gets the upper cutoff frequency of the filter.

        Returns
        -------
        float
            The upper cutoff frequency of the filter.
        """
        return self.__highcut

    def set_highcut(self, highcut: float):
        """
        Sets the upper cutoff frequency of the filter.

        Parameters
        ----------
        highcut : float
            The upper cutoff frequency of the filter.
        """
        self.__highcut = highcut

    def get_order(self) -> int:
        """
        Gets the order of the filter.

        Returns
        -------
        int
            The order of the filter.
        """
        return self.__order

    def set_order(self, order: int):
        """
        Sets the order of the filter.

        Parameters
        ----------
        order : int
            The order of the filter.
        """
        self.__order = order

    def get_analog(self) -> bool:
        """
        Gets the analog filter property of the filter.

        Returns
        -------
        bool
            True if the filter is an analog filter, False if it is a digital filter.
        """
        return self.__analog

    def set_analog(self, analog: bool):
        """
        Sets the analog filter property of the filter.

        Parameters
        ----------
        analog : bool
            True if the filter is an analog filter, False if it is a digital filter.
        """
        self.__analog = analog

    def apply_filter(self, signal: np.ndarray, sample_rate) -> np.ndarray:
        """
        Applies the band-stop filter to the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The signal to which the filter is to be applied.
        sample_rate : int
            The sample rate of the signal.

        Returns
        -------
        np.ndarray
            The filtered signal.
        """
        return self.__butter_bandstop_filter(signal, sample_rate)

    def __butter_bandstop(self, lowcut: float, highcut: float, sample_rate: int, order: int, analog: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the coefficients for a Butterworth band-stop filter.

        Parameters
        ----------
        lowcut : float
            The lower cutoff frequency of the filter.
        highcut : float
            The upper cutoff frequency of the filter.
        sample_rate : int
            The sample rate of the signal.
        order : int
            The order of the filter.
        analog : bool
            Whether the filter is an analog filter, or a digital filter.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The coefficients for the filter.
        """
        # Calculate nyquist frequency
        nyquist = sample_rate / 2
        # Calculate normalized frequencies
        low = lowcut / nyquist
        high = highcut / nyquist
        # Calculate the coefficients for the filter
        b, a = butter(order, [low, high], btype='bandstop', analog=analog, output='ba')
        return b, a

    def __butter_bandstop_filter(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies the Butterworth band-stop filter to the given data.

        Parameters
        ----------
        data : np.ndarray
            The data to which the filter is to be applied.
        sample_rate : int
            The sample rate of the data.

        Returns
        -------
        np.ndarray
            The filtered data.
        """
        # Calculate the coefficients for the filter
        b, a = self.__butter_bandstop(self.__lowcut, self.__highcut, sample_rate, self.__order, self.__analog)
        # Apply the filter
        y = filtfilt(b, a, data)
        return y
