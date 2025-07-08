import numpy as np
import scipy

from src.FeatureEngineering.Spectrograms.Interface import FrequencySpectrumCalculator


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
