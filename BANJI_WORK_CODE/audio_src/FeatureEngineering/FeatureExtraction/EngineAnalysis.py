from typing import List

import numpy as np
#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite

from src.DataIO.Representation.Datum import Datum, RawDatumKey, DerivedDataKey
from src.FeatureEngineering.FeatureExtraction.Interface import FeatureExtraction
from src.FeatureEngineering.Filters.Decimate import Resample
from src.FeatureEngineering.Filters.Other import TunableFilter
from src.FeatureEngineering.Spectrograms.Fourier import RFFT


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


class EngineState(FeatureExtraction):
    
    def __init__(self, model_path: str, label_literals: List[str]) -> None:
        """
        Initializes the engine state feature extraction class.

        This class uses a pre-trained TFLite model to predict the engine state of the given signal.

        Args:
            model_path: The path to the pre-trained TFLite model.
            label_literals: A list of strings representing the possible engine states.
        """
        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input/output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Store the label literals
        self.label_literals = label_literals
        # Initialize the RFFT calculator
        self.rfft_calculator = RFFT()
        # Initialize the buffer to an empty array
        self.__buffer = np.array([])
        # Store the buffer size in seconds
        self.__buffer_seconds = 1
        # Initialize the buffer results to an empty array
        self.__buffer_results = []
        # Store the buffer results dimension
        self.__buffer_results_dimension = 70

    def extract_features(self, signal: Datum) -> Datum:
        """
        Extracts features from the given signal.

        This feature extraction class uses a pre-trained TFLite model to predict the engine state of the given signal.

        Args:
            signal: The signal to extract features from.

        Returns:
            The signal with the extracted features.
        """
        # Set the sampling rate of the RFFT calculator
        self.rfft_calculator.set_fs(signal.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE))

        # Get the decimated audio and its sample rate
        signal_array = signal.get_derived_data(DerivedDataKey.DECIMATED_AUDIO)
        sample_rate = signal.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE)

        # Append the current signal to the buffer and then trim it to the specified size
        self.__buffer = np.concatenate((self.__buffer, signal_array), axis=None)
        self.__buffer = self.__buffer[-int(sample_rate * self.__buffer_seconds):]

        # If the buffer is full, calculate the RFFT representation of the buffer, predict the engine state using the pre-trained model
        # and add the predicted engine state to the signal's derived data
        if len(self.__buffer) == int(sample_rate * self.__buffer_seconds):
            rfft_repr = self.rfft_calculator.make_spectrum(self.__buffer)
            amplitude_spectrum = np.array([np.abs(rfft_repr)], dtype=np.float32)

            # Set input tensor for TFLite model
            self.interpreter.set_tensor(self.input_details[0]['index'], amplitude_spectrum)

            # Run inference
            self.interpreter.invoke()

            # Get prediction results
            pred_vector = self.interpreter.get_tensor(self.output_details[0]['index'])
            pred = np.argmax(pred_vector)
        else:
            # If the buffer is not full, set the engine state to unknown
            pred = 1

        # Append the predicted engine state to the buffer results
        self.__buffer_results.append(pred)
        # Trim the buffer results to the specified size
        self.__buffer_results = self.__buffer_results[-int(self.__buffer_results_dimension):]
        # Predict the engine state by taking the mode of the buffer results
        pred = np.argmax(np.bincount(self.__buffer_results))
        # Add the predicted engine state to the signal's derived data
        signal.add_derived_data(DerivedDataKey.ENGINE_STATE, self.label_literals[pred])
        return signal

