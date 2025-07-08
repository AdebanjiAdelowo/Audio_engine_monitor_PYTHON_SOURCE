import numpy as np
import pyaudio

from src.DataIO.Communication.Interface import CommunicationInterface
from src.DataIO.Representation.Datum import Datum, DatumType, RawDatumKey


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
