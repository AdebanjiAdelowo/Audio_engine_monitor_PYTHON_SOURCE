import numpy as np
import scipy

from src.FeatureEngineering.Filters.Butterworth import BandPassFilter
from src.FeatureEngineering.Filters.Interface import Filter
from src.FeatureEngineering.Spectrograms.Fourier import FFT


class PowerFilter(Filter):

    """
    Filter out the signal by setting a power threshold.

    Signals with values below the power threshold are set to zero.
    """
    def __init__(self, order: int = 3) -> None:
        """
        Parameters
        ----------
        order : int
            The order of the filter. Default is 3.

        """
        self.__order = order

    def get_order(self) -> int:
        """
        Gets the order of the PowerFilter.

        Returns
        -------
        int
            The order of the PowerFilter.

        """
        return self.__order

    def set_order(self, order: int):
        """
        Sets the order of the PowerFilter.

        Parameters
        ----------
        order : int
            The order of the PowerFilter.

        """
        self.__order = order

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply the PowerFilter to the given signal.

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
        # Calculate the absolute value of the signal
        a = np.abs(signal)

        # Calculate the average of the absolute values
        avg = scipy.stats.tmean(a)

        # Calculate the standard deviation of the absolute values
        std = self.__order * np.std(a)

        # Apply the filter
        filtered = (a > (avg + std)) * signal

        return filtered


class TunableFilter(Filter):

    def __init__(self, init_freq: float = 125, bandwidth: float = 50, order: int = 2) -> None:
        """
        Initialize the TunableFilter.

        Parameters
        ----------
        init_freq : float
            The initial frequency of the filter.
        bandwidth : float
            The bandwidth of the filter.
        order : int
            The order of the filter.

        """
        # The initial frequency of the filter
        self.__peak_freq = init_freq

        # The bandwidth of the filter
        self.__bandwidth = bandwidth

        # The order of the filter
        self.__order = order

        # The band-pass filter
        self.__bp_filter = BandPassFilter(
            # The lower cutoff frequency
            lowcut=10,
            # The upper cutoff frequency
            highcut=200,
            # The order of the filter
            order=self.__order,
            # Analog or digital filter
            analog=False
        )

        # The FFT calculator
        self.__fft = None

    def get_peak_freq(self) -> float:
        """
        Get the peak frequency of the filter.

        Returns
        -------
        float
            The peak frequency of the filter.

        """
        return self.__peak_freq

    def set_peak_freq(self, freq: float):
        """
        Set the peak frequency of the filter.

        Parameters
        ----------
        freq : float
            The new peak frequency.

        """
        self.__peak_freq = freq

        # If the frequency is too low, set the lower cutoff to 10 Hz
        # and the upper cutoff to 150 Hz
        # Otherwise, set the lower and upper cutoffs based on the
        # bandwidth
        if self.__peak_freq - self.__bandwidth / 2 < 20:
            self.__bp_filter.set_lowcut(10)
        else:
            self.__bp_filter.set_lowcut(max(freq - self.__bandwidth / 2, 20))

        if self.__peak_freq + self.__bandwidth / 2 > 150:
            self.__bp_filter.set_highcut(150)
        else:
            self.__bp_filter.set_highcut(min(freq + self.__bandwidth / 2, 150))

    def get_bandwidth(self) -> float:
        """
        Get the bandwidth of the filter.

        Returns
        -------
        float
            The bandwidth of the filter.

        """
        return self.__bandwidth

    def set_bandwidth(self, bandwidth: float):
        """
        Set the bandwidth of the filter.

        Parameters
        ----------
        bandwidth : float
            The new bandwidth.

        """
        # Store the bandwidth
        self.__bandwidth = bandwidth

        # Set the lower and upper cutoffs based on the bandwidth
        self.__bp_filter.set_lowcut(self.__peak_freq - self.__bandwidth / 2)
        self.__bp_filter.set_highcut(self.__peak_freq + self.__bandwidth / 2)

    def get_order(self) -> int:
        """
        Get the order of the filter.

        Returns
        -------
        int
            The order of the filter.

        """
        return self.__order

    def set_order(self, order: int):
        """
        Set the order of the filter.

        Parameters
        ----------
        order : int
            The new order.

        """
        # Store the order
        self.__order = order

        # Set the order of the BandPassFilter
        self.__bp_filter.set_order(order)

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply the filter to the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The signal to filter.
        sample_rate : int
            The sample rate of the signal.

        Returns
        -------
        np.ndarray
            The filtered signal.

        """
        # Remove the mean from the signal
        signal = signal - np.mean(signal)

        # Apply the band-pass filter
        filtered_signal = self.__bp_filter.apply_filter(signal, sample_rate)

        # Create a FFT calculator if it does not exist
        if self.__fft is None:
            self.__fft = FFT(sample_rate)
        # Reset the FFT calculator if the sample rate has changed
        elif self.__fft.get_fs() != sample_rate:
            self.__fft.set_fs(sample_rate)

        # Make the spectrum of the filtered signal
        freq_signal = self.__fft.make_spectrum(filtered_signal)

        # Calculate the power spectrum
        power_spectrum = np.abs(freq_signal[:len(freq_signal) // 2])

        # Find the peaks in the power spectrum
        peak_indices = scipy.signal.find_peaks(power_spectrum, distance=sample_rate)[0]
        peak_frequencies = self.__fft.get_freq()[peak_indices]

        # If there are no peaks, return the filtered signal
        if len(peak_indices) == 0:
            return filtered_signal

        # Find the minimum peak index
        peak_index = peak_indices.min()

        # Find the frequency of the minimum peak
        peak_frequency = self.__fft.get_freq()[peak_index]

        # Set the peak frequency
        self.set_peak_freq(peak_frequency)

        # Return the filtered signal
        return filtered_signal
