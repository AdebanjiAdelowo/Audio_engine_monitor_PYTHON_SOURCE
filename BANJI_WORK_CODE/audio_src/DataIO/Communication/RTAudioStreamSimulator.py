from typing import Optional
from pydub import AudioSegment

# import librosa
import numpy as np

from src.DataIO.Communication.Interface import CommunicationInterface
from src.DataIO.Representation.Datum import Datum, DatumType, RawDatumKey


class _RTAudioSimulator:
    def __init__(self, audio_file: str, buffer_seconds: float = 1) -> None:
        """
        Initialize the _RTAudioSimulator object.

        :param audio_file: The path to the audio file
        :param buffer_seconds: The buffer size in seconds
        """
        # Load the audio file
        # audio, sample_rate = librosa.load(audio_file, sr = None)
        audio = AudioSegment.from_file(audio_file)
        
        # Store the loaded audio in the object
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) # np.array(audio)
        samples  /= np.iinfo(np.int16).max
        
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)
            
        self.__audio_array  = samples
        
        sample_rate = audio.frame_rate  # Get sample rate
        
        # Store the sample rate of the audio
        self.__sample_rate = sample_rate
        # Compute the buffer size in samples
        self.__buffer_dim = int(buffer_seconds * sample_rate)

    def get_audio_array(self) -> np.ndarray:
        """
        Returns the audio array.

        :return: The audio array
        """
        return self.__audio_array

    def get_sample_rate(self) -> int:
        """
        Returns the sample rate of the audio.

        :return: The sample rate of the audio
        """
        return self.__sample_rate

    def __iter__(self) -> iter:
        """
        Iterate over the audio array in chunks of length self.__buffer_dim.

        This method is used to simulate the audio stream by yielding chunks of
        the audio array.

        :return: An iterator over the audio array
        """
        # Iterate over the audio array in chunks of length self.__buffer_dim
        for i in range(self.__buffer_dim, len(self.__audio_array), self.__buffer_dim):
            # Extract the current chunk of the audio array
            a = self.__audio_array[i - self.__buffer_dim:i]
            # Yield the current chunk
            yield a


class RTAudioStreamSimulator(CommunicationInterface):
    def __init__(self, audio_file: str, buffer_seconds: float = 1) -> None:
        """
        Initialize the RTAudioStreamSimulator object.

        :param audio_file: The path to the audio file
        :param buffer_seconds: The buffer size in seconds
        """
        self.__stream = None  # The simulated audio stream
        self.__stream_iterator = None  # The iterator over the simulated audio stream
        self.__audio_file = audio_file  # The path to the audio file
        self.__buffer_seconds = buffer_seconds  # The buffer size in seconds

    def start(self) -> None:
        """
        Start the simulated audio stream.

        This method creates a _RTAudioSimulator object and an iterator over the
        simulated audio stream.
        """
        # Create a _RTAudioSimulator object
        self.__stream = _RTAudioSimulator(self.__audio_file, buffer_seconds=self.__buffer_seconds)
        # Create an iterator over the simulated audio stream
        self.__stream_iterator = iter(self.__stream)

    def read_sample(self) -> Optional[Datum]:
        """
        Read a sample from the simulated audio stream.

        This method reads a chunk of the audio array from the simulated audio
        stream and creates a Datum object with the audio array and the sample
        rate.

        :return: A Datum object with the audio array and the sample rate or None
                 if the end of the audio stream is reached
        """
        try:
            # Read a chunk of the audio array from the simulated audio stream
            audio_array = next(self.__stream_iterator)
        except StopIteration:
            # If the end of the audio stream is reached, return None
            return None
        # Create a Datum object with the audio array and the sample rate
        datum = Datum(DatumType.AUDIO)
        datum.add_raw_datum(RawDatumKey.AUDIO_ARRAY, audio_array)
        datum.add_raw_datum(RawDatumKey.SAMPLE_RATE, self.__stream.get_sample_rate())
        # Return the Datum object
        return datum
    

    def write_sample(self) -> Optional[Datum]:
        """
        Read a sample from the simulated audio stream.

        This method reads a chunk of the audio array from the simulated audio
        stream and creates a Datum object with the audio array and the sample
        rate.

        :return: A Datum object with the audio array and the sample rate or None
                 if the end of the audio stream is reached
        """
        pass

    def close(self):
        """
        Close the simulated audio stream.

        This method closes the simulated audio stream and releases all the
        resources allocated by the stream.
        """
        # Close the simulated audio stream
        self.__stream = None
        self.__stream_iterator = None
