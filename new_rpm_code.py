from enum import Enum
import numpy as np
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import os
import io
import json
import matplotlib.pyplot as plt
import pyaudio
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from scipy.signal import resample
from scipy.fft import rfft
from typing import List
from abc import ABC, abstractmethod

import numpy as np
from scipy import signal


class DatumType(Enum):
    NUMERIC = 1
    AUDIO = 2
    IMAGE = 3
    VIDEO = 4
    TEXT = 5


class RawDatumKey(Enum):
    AUDIO_ARRAY = 'audio_array'
    SAMPLE_RATE = 'sample_rate'


class DerivedDataKey(Enum):
    ENGINE_STATE = 'engine_state'
    DECIMATED_AUDIO_RATE = 'decimated_audio_rate'
    RPM = 'RPM'
    DECIMATED_AUDIO = 'decimated_audio'
    FREQUENCY_PEAK = 'frequency_peak'
    FREQUENCY_PEAK_FILTERED_AUDIO = 'frequency_peak_filtered_audio'
    AMPLITUDE_SPECTRUM = 'amplitude_spectrum'


class Datum:
    def __init__(self, datum_type: DatumType) -> None:
        """
        :param datum_type: The type of the datum
        """
        self.__raw_datum: Dict[RawDatumKey, Any] = {}
        """
        A dictionary containing the raw data of the Datum.
        The keys are the RawDatumKey and the values are the raw data.
        """
        self.__datum_type: DatumType = datum_type
        """
        The type of the Datum.
        """
        self.__derived_data: Optional[Dict[DerivedDataKey, Any]] = {}
        """
        A dictionary containing the derived data of the Datum.
        The keys are the DerivedDataKey and the values are the derived data.
        """
        self.__labels: Optional[List] = []
        """
        A list of the labels of the Datum.
        """
        self.__labels_names: Optional[List[str]] = []
        """
        A list of the names of the labels of the Datum.
        """

    def add_raw_datum(self, key: RawDatumKey, value: Any) -> None:
        """
        Adds a raw datum to the Datum.

        :param key: The key of the raw datum.
        :param value: The value of the raw datum.
        """
        self.__raw_datum[key] = value

    def remove_raw_datum(self, key: RawDatumKey) -> None:
        """
        Removes a raw datum from the Datum.

        :param key: The key of the raw datum to be removed.
        """
        del self.__raw_datum[key]

    def get_raw_datum(self, key: RawDatumKey) -> Any:
        """
        Gets a raw datum from the Datum.

        :param key: The key of the raw datum.
        :return: The value of the raw datum.
        """
        return self.__raw_datum[key]

    def get_raw_datum_keys(self) -> List[RawDatumKey]:
        """
        Gets a list of all the raw data keys of the Datum.

        :return: A list of RawDatumKey.
        """
        return list(self.__raw_datum.keys())

    def add_derived_data(self, key: DerivedDataKey, value: Any) -> None:
        """
        Adds a derived datum to the Datum.

        :param key: The key of the derived datum.
        :param value: The value of the derived datum.
        """
        self.__derived_data[key] = value

    def remove_derived_data(self, key: DerivedDataKey) -> None:
        """
        Removes a derived datum from the Datum.

        :param key: The key of the derived datum to be removed.
        """
        del self.__derived_data[key]

    def get_derived_data(self, key: DerivedDataKey) -> Any:
        """
        Gets a derived datum from the Datum.

        :param key: The key of the derived datum.
        :return: The value of the derived datum.
        """
        return self.__derived_data[key]

    def get_derived_data_keys(self) -> List[DerivedDataKey]:
        """
        Gets a list of all the derived data keys of the Datum.

        :return: A list of DerivedDataKey.
        """
        return list(self.__derived_data.keys())

    def set_labels(self, labels: List) -> None:
        """
        Sets the labels of the Datum.

        :param labels: The labels to be set.
        """
        self.__labels = labels

    def set_labels_names(self, labels_names: List[str]) -> None:
        """
        Sets the names of the labels of the Datum.

        :param labels_names: The names of the labels to be set.
        """
        self.__labels_names = labels_names

    def get_labels(self) -> List:
        """
        Gets the labels of the Datum.

        :return: A list of the labels of the Datum.
        """
        return self.__labels

    def get_labels_names(self) -> List[str]:
        """
        Gets the names of the labels of the Datum.intuos04
        

        :return: A list of strings representing the names of the labels.
        """
        return self.__labels_names



class Filter(ABC):

    @abstractmethod
    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies the filter to the given signal.

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
        pass


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
    
class STFT(FrequencySpectrumCalculator):

    def __init__(self, fs=44100, window='hann', nperseg=None, noverlap=None, nfft=None, scaling='spectrum'):
        """
        Initialize the STFT (Short-Time Fourier Transform) calculator.

        Parameters
        ----------
        fs : int
            The sampling frequency of the time series.
        window : str
            The type of window used for the STFT.
        nperseg : int
            The number of samples per segment.
        noverlap : int
            The number of samples to overlap between segments.
        nfft : int
            The number of frequency bins to use for the FFT.
        scaling : str
            The type of scaling used for the FFT.
        """
        self.__fs = fs
        self.__window = window
        self.__nperseg = nperseg
        self.__noverlap = noverlap
        self.__nfft = nfft
        self.__scaling = scaling

        self.__time = None
        self.__freq = None

    def get_time(self):
        """
        Returns the time array for the STFT calculation.

        Returns
        -------
        time : array_like
            The time array for the STFT calculation.
        """
        return self.__time

    def get_freq(self):
        """
        Returns the frequency array for the STFT calculation.

        Returns
        -------
        freq : array_like
            The frequency array for the STFT calculation.
        """
        return self.__freq

    def make_spectrum(self, time_series: np.ndarray):
        """
        Calculates the STFT of the given time series.

        Parameters
        ----------
        time_series : array_like
            The time series to calculate the STFT of.

        Returns
        -------
        stft : array_like
            The STFT of the time series.
        """
        # Calculate the STFT
        f, t, Zxx = scipy.signal.stft(time_series, fs=self.__fs, window=self.__window, nperseg=self.__nperseg,
                                noverlap=self.__noverlap, nfft=self.__nfft, scaling=self.__scaling)

        # Store the time and frequency arrays
        self.__time = t
        self.__freq = f

        return Zxx

    def make_timeseries(self, spectrogram: np.ndarray):
        """
        Calculates the inverse STFT of the given spectrogram.

        Parameters
        ----------
        spectrogram : array_like
            The spectrogram to calculate the inverse STFT of.

        Returns
        -------
        time_series : array_like
            The time series calculated from the spectrogram.
        """
        # Calculate the inverse STFT
        time_series, _ = scipy.signal.istft(spectrogram, fs=self.__fs, window=self.__window, nperseg=self.__nperseg,
                                     noverlap=self.__noverlap, nfft=self.__nfft, scaling=self.__scaling)

        return time_series


class FFT(FrequencySpectrumCalculator):

    def __init__(self, fs=44100, nfft=None, scaling='spectrum'):
        """
        Initializes the FFT calculator.

        Parameters
        ----------
        fs : int
            The sampling frequency of the time series.
        nfft : int
            The number of frequency bins in the FFT.
        scaling : str
            The type of scaling used in the FFT.
        """
        self.__fs = fs
        self.__nfft = nfft
        self.__scaling = scaling

        # The frequency array will be stored here
        self.__freq = None

    def get_fs(self):
        """
        Returns the sampling frequency of the FFT calculator.

        Returns
        -------
        fs : int
            The sampling frequency of the FFT calculator.
        """
        return self.__fs

    def set_fs(self, fs: int):
        """
        Sets the sampling frequency of the FFT calculator.

        Parameters
        ----------
        fs : int
            The sampling frequency of the FFT calculator.
        """
        self.__fs = fs

    def get_freq(self):
        """
        Returns the frequency array calculated by the FFT calculator.

        Returns
        -------
        freq : array_like
            The frequency array calculated by the FFT calculator.
        """
        return self.__freq

    def make_spectrum(self, time_series: np.ndarray):
        """
        Calculates the FFT of the given time series.

        Parameters
        ----------
        time_series : array_like
            The time series to calculate the FFT of.

        Returns
        -------
        yf : array_like
            The FFT of the time series.
        """
        # Calculate the FFT
        yf = scipy.fft.fft(time_series, n=self.__nfft)

        # Calculate the frequency array
        self.__freq = scipy.fft.fftfreq(len(time_series), 1 / self.__fs)

        return yf

    def make_timeseries(self, spectrum: np.ndarray):
        """
        Calculates the inverse FFT of the given spectrum.

        Parameters
        ----------
        spectrum : array_like
            The spectrum to calculate the inverse FFT of.

        Returns
        -------
        time_series : array_like
            The inverse FFT of the spectrum.
        """
        # Calculate the inverse FFT
        return scipy.ifft(spectrum, n=self.__nfft)
    

class RFFT(FrequencySpectrumCalculator):

    def __init__(self, fs=44100, nfft=None, scaling='spectrum'):
        """
        Initializes the RFFT calculator.
        

        Parameters
        ----------
        fs : int
            The sampling frequency of the time series.
        nfft : int
            The number of frequency bins in the RFFT.
        scaling : str
            The type of scaling used in the RFFT.
        """
        self.__fs = fs
        self.__nfft = nfft
        self.__scaling = scaling

        # The frequency array will be stored here
        self.__freq = None

    def get_fs(self):
        """
        Returns the sampling frequency of the RFFT calculator.

        Returns
        -------
        fs : int
            The sampling frequency of the RFFT calculator.
        """
        return self.__fs

    def set_fs(self, fs):
        """
        Sets the sampling frequency of the RFFT calculator.

        Parameters
        ----------
        fs : int
            The sampling frequency of the RFFT calculator.
        """
        self.__fs = fs

    def get_freq(self):
        """
        Returns the frequency array calculated by the RFFT calculator.

        Returns
        -------
        freq : array_like
            The frequency array calculated by the RFFT calculator.
        """
        return self.__freq

    def make_spectrum(self, time_series: np.ndarray):
        """
        Calculates the RFFT of the given time series.

        Parameters
        ----------
        time_series : array_like
            The time series to calculate the RFFT of.

        Returns
        -------
        yf : array_like
            The RFFT of the time series.
        """
        # Normalize the time series to prevent overflow
        time_series_normalized = time_series / np.max(np.abs(time_series))

        # Calculate the RFFT
        yf = scipy.fft.rfft(time_series_normalized, n=self.__nfft)[1:]

        # Calculate the frequency array
        self.__freq = scipy.fft.rfftfreq(len(time_series), 1 / self.__fs)[1:]

        # Normalize the result to prevent overflow
        yf_norm = yf / np.max(np.abs(yf))

        return yf_norm

    def make_timeseries(self, spectrum: np.ndarray):
        """
        Calculates the inverse RFFT of the given spectrum.

        Parameters
        ----------
        spectrum : array_like
            The spectrum to calculate the inverse RFFT of.

        Returns
        -------
        time_series : array_like
            The inverse RFFT of the spectrum.
        """
        # Calculate the inverse RFFT
        return scipy.fft.irfft(spectrum, n=self.__nfft)

    
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
    
class CommunicationInterface(ABC):

    @abstractmethod
    def read_sample(self) -> bytes:
        """Reads a sample from the communication interface.

        Returns:
            bytes: The sample read from the communication interface.
        """
        pass

    @abstractmethod
    def write_sample(self, sample: bytes) -> None:
        """Writes a sample to the communication interface.

        Args:
            sample (bytes): The sample to be written to the communication interface.
        """
        pass


class FeatureEngineeringPipeline:

    def __init__(self):
        """
        Initializes a new feature engineering pipeline.

        The feature engineering pipeline is a sequence of feature extraction blocks
        that are applied to a given datum to produce the final output.
        """
        # The feature engineering blocks are the main fea
        # ture extraction blocks
        # that are used to extract meaningful features from the given datum.
        self.__feature_engineering_blocks = []

        # The output blocks are the blocks that are used to post-process the
        # output of the feature engineering blocks and produce the final output.
        self.__output_blocks = []

    def add_block(self, block: FeatureExtraction) -> None:
        """
        Adds a feature engineering block to the pipeline.

        The block is added to the feature engineering blocks list
        and is executed in the order it was added.

        Args:
            block: The feature engineering block to add.
        """
        self.__feature_engineering_blocks.append(block)

    def add_output_block(self, block: FeatureExtraction) -> None:
        """
        Adds a output block to the pipeline.

        The output blocks are executed after the feature engineering blocks
        and are used to post-process the output of the feature engineering
        blocks and produce the final output.

        Args:
            block: The output block to add.
        """
        self.__output_blocks.append(block)

    def run(self, datum: Datum) -> Datum:
        """
        Runs the feature engineering pipeline.

        The pipeline is composed of a sequence of feature engineering blocks
        that are applied to the given datum to produce the final output.

        Args:
            datum: The datum to process.

        Returns:
            The processed datum.
        """
        # Run the feature engineering blocks
        for block in self.__feature_engineering_blocks:
            # Each block is executed in the order it was added
            datum = block.extract_features(datum)

        # Run the output blocks
        for block in self.__output_blocks:
            # The output blocks are executed after the feature engineering blocks
            # and are used to post-process the output of the feature engineering
            # blocks and produce the final output
            datum = block.extract_features(datum)

        return datum


class AudioStream(CommunicationInterface):

    def __init__(self, fs: int = 44100,
                 buffer_seconds: float = 1,
                 channels: int = 1,
                 audio_format=pyaudio.paInt16,
                 audio_input: bool = True,
                 audio_output: bool = False):
        """
        Initialize an AudioStream.
        

        Parameters
        ----------
        fs : int
            Sample rate in Hz. Default is 44100.
        buffer_seconds : float
            Size of the buffer in seconds. Default is 1.
        channels : int
            Number of channels. Default is 1.
        audio_format : int
            Sample format. Default is pyaudio.paInt16.
        audio_input : bool
            If True, the stream is opened for audio input. Default is True.
        audio_output : bool
            If True, the stream is opened for audio output. Default is False.
        """
        self.__p = pyaudio.PyAudio()
        self.__fs = fs
        self.__buffer_length = int(buffer_seconds * fs)
        self.__channels = channels
        self.__audio_format = audio_format
        self.__audio_input = audio_input
        self.__audio_output = audio_output

        self.__stream = None

    def start(self):
        """
        Open the stream for audio input/output.

        This method opens the PyAudio stream according to the parameters set during initialization.
        It must be called before reading or writing samples.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.__stream = self.__p.open(format=self.__audio_format,
                                      channels=self.__channels,
                                      rate=self.__fs,
                                      input=self.__audio_input,
                                      output=self.__audio_output,
                                      frames_per_buffer=self.__buffer_length
                                      )

    def read_sample(self) -> Datum:
        """
        Read an audio sample from the stream.

        This method reads a sample from the audio stream and returns it as a Datum.

        Parameters
        ----------
        None

        Returns
        -------
        sample : Datum
            The read audio sample as a Datum with type DatumType.AUDIO.
            The Datum contains the raw audio data as numpy array in the RawDatumKey.AUDIO_ARRAY key and the sample rate in the RawDatumKey.SAMPLE_RATE key.
        """
        dtype = np.int16
        audio = np.frombuffer(self.__stream.read(self.__buffer_length), dtype=dtype)
        datum = Datum(DatumType.AUDIO)
        datum.add_raw_datum(RawDatumKey.AUDIO_ARRAY, audio)
        datum.add_raw_datum(RawDatumKey.SAMPLE_RATE, self.__fs)
        return datum
    
    def write_sample(self) -> Datum:
        pass
    
    def close(self):
        """
        Close the stream and terminate the PyAudio interface.

        This method stops the stream, closes it and terminates the PyAudio interface.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.__stream.stop_stream()
        self.__stream.close()
        self.__p.terminate()







class Decimation(FeatureExtraction):

    """
    A feature extraction class that downsamples the input signal to a target sampling rate.
    """
    def __init__(self, target_rate: int = 500) -> None:
        """
        Initializes the decimation feature extraction class.

        Args:
            target_rate: The target sampling rate. Defaults to 500.
        """
        self.__filter = Resample(new_rate=target_rate)

    def extract_features(self, signal: Datum) -> Datum:
        """
        Extracts features from the given signal.

        This feature extraction class downsamples the input signal to a target sampling rate.

        Args:
            signal: The signal to extract features from.

        Returns:
            The signal with the extracted features.
        """
        # Apply the decimation filter to the signal
        filtered_signal = self.__filter.apply_filter(
                                                     signal.get_raw_datum(RawDatumKey.AUDIO_ARRAY),
                                                     signal.get_raw_datum(RawDatumKey.SAMPLE_RATE)
                                                    )

        # Add the decimated signal to the signal's derived data
        signal.add_derived_data(DerivedDataKey.DECIMATED_AUDIO, filtered_signal)

        # Add the decimated sampling rate to the signal's derived data
        signal.add_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, self.__filter.get_new_rate())

        return signal


class FrequencyPeakFinder(FeatureExtraction):

    def __init__(self, buffer_seconds: float = 10) -> None:
        """
        Initializes the frequency peak finder feature extraction class.

        This feature extraction class finds the frequency peak in the given signal.

        Args:
            buffer_seconds: The number of seconds worth of data to keep in the buffer. Defaults to 10.
        """
        self.__filter = TunableFilter(init_freq=100, bandwidth=10, order=4)
        # Initialize the buffer to an empty array
        self.__buffer = np.array([])
        # Store the buffer size in seconds
        self.__buffer_seconds = buffer_seconds

    def extract_features(self, signal: Datum) -> Datum:
        """
        Extracts features from the given signal.

        This feature extraction class applies a bandpass filter to the signal and then finds the frequency
        peak in the filtered signal.

        Args:
            signal: The signal to extract features from.

        Returns:
            The signal with the extracted features.
        """

        # If the signal has already been decimated, use the decimated signal
        if DerivedDataKey.DECIMATED_AUDIO in signal.get_derived_data_keys():
            signal_array = signal.get_derived_data(DerivedDataKey.DECIMATED_AUDIO)
            sample_rate = signal.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE)
        # Otherwise, use the raw audio
        else:
            signal_array = signal.get_raw_datum(RawDatumKey.AUDIO_ARRAY)
            sample_rate = signal.get_raw_datum(RawDatumKey.SAMPLE_RATE)

        # Append the current signal to the buffer and then trim it to the specified size
        self.__buffer = np.concatenate((self.__buffer, signal_array), axis=None)
        self.__buffer = self.__buffer[-int(sample_rate * self.__buffer_seconds):]

        # Apply the filter to the buffer and add the filtered signal to the signal's derived data
        filtered_signal = self.__filter.apply_filter(self.__buffer, sample_rate)
        signal.add_derived_data(DerivedDataKey.FREQUENCY_PEAK_FILTERED_AUDIO, filtered_signal)

        # Add the frequency peak to the signal's derived data
        signal.add_derived_data(DerivedDataKey.FREQUENCY_PEAK, self.__filter.get_peak_freq())

        return signal


class RPM(FeatureExtraction):

    def __init__(self, events_per_cranckshaft_cycle: int) -> None:
        """
        Calculates the RPM of the engine given the frequency peak of the engine sound.

        Args:
            events_per_cranckshaft_cycle: The number of events per crankshaft cycle. This is used to calculate the RPM.
        """
        self.__events_per_cranckshaft_cycle = events_per_cranckshaft_cycle

    def extract_features(self, signal: Datum) -> Datum:
        """
        Extracts features from the given signal.

        This feature extraction class calculates the RPM of the engine given the frequency peak of the engine sound.

        Args:
            signal: The signal to extract features from.

        Returns:
            The signal with the extracted features.
        """
        # Get the frequency peak of the signal
        peak = signal.get_derived_data(DerivedDataKey.FREQUENCY_PEAK)

        # Calculate the RPM based on the frequency peak
        rpm = self.__calculate_rpm(peak)

        # Add the RPM to the signal's derived data
        signal.add_derived_data(DerivedDataKey.RPM, rpm)

        return signal

    def __calculate_rpm(self, peak: float) -> float:
        """
        Calculate the RPM based on the frequency peak of the engine sound.

        Args:
            peak: The frequency peak of the engine sound.

        Returns:
            The RPM of the engine.
        """
        # The RPM is calculated by multiplying the frequency peak with 60 and
        # dividing it by the number of events per crankshaft cycle.
        return peak * 60 / self.__events_per_cranckshaft_cycle
    
    
audio_buffer_seconds=1, fft_buffer_seconds=10
audio_stream = AudioStream(fs=8000, buffer_seconds=audio_buffer_seconds)
audio_stream.start()
pipeline = FeatureEngineeringPipeline()
pipeline.add_block(Decimation(target_rate=500))
pipeline.add_block(FrequencyPeakFinder(buffer_seconds=fft_buffer_seconds))
pipeline.add_output_block(RPM(events_per_cranckshaft_cycle=2))
# i = 0
audio_array = audio_stream.read_sample()
audio_data = pipeline.run(audio_array)




















