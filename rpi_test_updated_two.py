from abc import ABC, abstractmethod  # MISSING IMPORT - Added
from enum import Enum
import os
import time
import threading
import queue
import struct
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
from scipy.signal import decimate, find_peaks, filtfilt, butter, resample, stft, istft
import sounddevice as sd
import serial
import psutil
import logging
import pyaudio
import matplotlib.pyplot as plt
import json
import io
from typing import List, Optional, Dict, Any, Tuple

# Settings
RECORD_SECONDS = 5
SAMPLE_RATE = 44100
AUDIO_DIR = "./recordings"
UART_PORT = "/dev/serial0"  # Change to your ESP32 port (e.g., "/dev/ttyUSB0")
UART_BAUDRATE = 115200
CLEAN_INTERVAL_DAYS = 7
DECIMATED_RATE = 500
EVENTS_PER_CYCLE = 2
DISK_SPACE_THRESHOLD = 0.1
MIN_FREE_SPACE_GB = 1.0
ANALYZE_QUEUE_SIZE = 10
MIC_DEVICE = None
LOG_FILE = "audio_system.log"
AUDIO_MODE = "file"  # Options: "file" (sounddevice) or "stream" (pyaudio)
AUDIO_BUFFER_SECONDS = 1
FFT_BUFFER_SECONDS = 10

# UART Protocol Constants
START_BYTE_DATA = 0xAA
START_BYTE_RECORD = 0xAC
START_BYTE_SYNC = 0xAB
END_BYTE = 0x55
ACK_TIMEOUT = 2.0
MAX_RETRIES = 5
INTER_PACKET_DELAY = 0.05

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs(AUDIO_DIR, exist_ok=True)

# Queues
analyze_queue = queue.Queue(maxsize=ANALYZE_QUEUE_SIZE)

# Serial UART init
ser = None
uart_lock = threading.Lock()

def init_uart():
    global ser
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            ser = serial.Serial(
                UART_PORT,
                UART_BAUDRATE,
                timeout=1,
                rtscts=False,
                xonxoff=False,
                write_timeout=2.0
            )
            time.sleep(2)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            print(f"✓ UART connected on {UART_PORT} at {UART_BAUDRATE} baud")
            logging.info(f"UART connected on {UART_PORT} at {UART_BAUDRATE} baud")
            return True
        except Exception as e:
            print(f"✗ UART connection failed (attempt {attempt + 1}/{max_attempts}): {e}")
            logging.error(f"UART connection failed: {e}")
            ser = None
            time.sleep(5)
    print("✗ Warning: Could not initialize UART. System will continue without UART.")  # FIXED - Changed from exit(1)
    logging.warning("Could not initialize UART. System will continue without UART.")
    return False  # FIXED - Return False instead of exiting

file_counter = 1
last_cleanup = time.time()
packet_seq_num = 0

# Enums
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

# Datum Class
class Datum:
    def __init__(self, datum_type: DatumType, audio_array=None, sample_rate=None):
        self.__raw_datum: Dict[RawDatumKey, Any] = {}
        self.__datum_type: DatumType = datum_type
        self.__derived_data: Dict[DerivedDataKey, Any] = {}
        self.__labels: Optional[List] = []
        self.__labels_names: Optional[List[str]] = []
        if audio_array is not None:
            self.__raw_datum[RawDatumKey.AUDIO_ARRAY] = audio_array
        if sample_rate is not None:
            self.__raw_datum[RawDatumKey.SAMPLE_RATE] = sample_rate

    def add_raw_datum(self, key: RawDatumKey, value: Any) -> None:
        self.__raw_datum[key] = value

    def remove_raw_datum(self, key: RawDatumKey) -> None:
        del self.__raw_datum[key]

    def get_raw_datum(self, key: RawDatumKey) -> Any:
        return self.__raw_datum.get(key)

    def get_raw_datum_keys(self) -> List[RawDatumKey]:
        return list(self.__raw_datum.keys())

    def add_derived_data(self, key: DerivedDataKey, value: Any) -> None:
        self.__derived_data[key] = value

    def remove_derived_data(self, key: DerivedDataKey) -> None:
        del self.__derived_data[key]

    def get_derived_data(self, key: DerivedDataKey) -> Any:
        return self.__derived_data.get(key)

    def get_derived_data_keys(self) -> List[DerivedDataKey]:
        return list(self.__derived_data.keys())

    def set_labels(self, labels: List) -> None:
        self.__labels = labels

    def set_labels_names(self, labels_names: List[str]) -> None:
        self.__labels_names = labels_names

    def get_labels(self) -> List:
        return self.__labels

    def get_labels_names(self) -> List[str]:
        return self.__labels_names

# Filter Classes
class Filter(ABC):  # FIXED - Now properly inherits from ABC
    @abstractmethod
    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        pass

class BandPassFilter(Filter):
    def __init__(self, lowcut: float, highcut: float, order: int, analog: bool) -> None:
        self.__lowcut = lowcut
        self.__highcut = highcut
        self.__order = order
        self.__analog = analog

    def get_lowcut(self) -> float:
        return self.__lowcut

    def set_lowcut(self, lowcut: float):
        self.__lowcut = lowcut

    def get_highcut(self) -> float:
        return self.__highcut

    def set_highcut(self, highcut: float):
        self.__highcut = highcut

    def get_order(self) -> int:
        return self.__order

    def set_order(self, order: int):
        self.__order = order

    def get_analog(self) -> bool:
        return self.__analog

    def set_analog(self, analog: bool):
        self.__analog = analog

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.__butter_bandpass_filter(signal, sample_rate)

    def __butter_bandpass(self, lowcut: float, highcut: float, sample_rate: int, order: int, analog: bool) -> Tuple[np.ndarray, np.ndarray]:
        nyquist = sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band', analog=analog, output='ba')
        return b, a

    def __butter_bandpass_filter(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        b, a = self.__butter_bandpass(self.__lowcut, self.__highcut, sample_rate, self.__order, self.__analog)
        y = filtfilt(b, a, data)
        return y

class TunableFilter(Filter):
    def __init__(self, init_freq: float = 125, bandwidth: float = 50, order: int = 2) -> None:
        self.__peak_freq = init_freq
        self.__bandwidth = bandwidth
        self.__order = order
        self.__bp_filter = BandPassFilter(
            lowcut=10, highcut=200, order=self.__order, analog=False
        )
        self.__fft = None

    def get_peak_freq(self) -> float:
        return self.__peak_freq

    def set_peak_freq(self, freq: float):
        self.__peak_freq = freq
        if self.__peak_freq - self.__bandwidth / 2 < 20:
            self.__bp_filter.set_lowcut(10)
        else:
            self.__bp_filter.set_lowcut(max(freq - self.__bandwidth / 2, 20))
        if self.__peak_freq + self.__bandwidth / 2 > 150:
            self.__bp_filter.set_highcut(150)
        else:
            self.__bp_filter.set_highcut(min(freq + self.__bandwidth / 2, 150))

    def get_bandwidth(self) -> float:
        return self.__bandwidth

    def set_bandwidth(self, bandwidth: float):
        self.__bandwidth = bandwidth
        self.__bp_filter.set_lowcut(self.__peak_freq - self.__bandwidth / 2)
        self.__bp_filter.set_highcut(self.__peak_freq + self.__bandwidth / 2)

    def get_order(self) -> int:
        return self.__order

    def set_order(self, order: int):
        self.__order = order
        self.__bp_filter.set_order(order)

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        signal = signal - np.mean(signal)
        filtered_signal = self.__bp_filter.apply_filter(signal, sample_rate)
        if self.__fft is None:
            self.__fft = FFT(sample_rate)
        elif self.__fft.get_fs() != sample_rate:
            self.__fft.set_fs(sample_rate)
        freq_signal = self.__fft.make_spectrum(filtered_signal)
        power_spectrum = np.abs(freq_signal[:len(freq_signal) // 2])
        peak_indices = find_peaks(power_spectrum, distance=sample_rate//100)[0]  # FIXED - Added reasonable distance
        if len(peak_indices) == 0:
            return filtered_signal
        peak_index = peak_indices[np.argmax(power_spectrum[peak_indices])]  # FIXED - Find highest peak
        peak_frequency = abs(self.__fft.get_freq()[peak_index])  # FIXED - Use abs() to handle negative frequencies
        self.set_peak_freq(peak_frequency)
        return filtered_signal

class AntiAliasingDecimation(Filter):
    def __init__(self, downsampling_factor: int = 3, order: int = 8) -> None:
        self.__q = downsampling_factor
        self.__order = order

    def get_downsampling_factor(self) -> int:
        return self.__q

    def set_downsampling_factor(self, downsampling_factor: int):
        self.__q = downsampling_factor

    def get_order(self) -> int:
        return self.__order

    def set_order(self, order: int):
        self.__order = order

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        return decimate(signal, self.__q, n=self.__order)

class Resample(Filter):
    def __init__(self, new_rate: int) -> None:
        self.__new_rate = new_rate

    def get_new_rate(self) -> int:
        return self.__new_rate

    def set_new_rate(self, new_rate: int):
        self.__new_rate = new_rate

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        resampled_length = int(len(signal) * self.__new_rate / sample_rate)
        return resample(signal, resampled_length)

# Frequency Spectrum Calculators
class FrequencySpectrumCalculator(ABC):  # FIXED - Now properly inherits from ABC
    @abstractmethod
    def make_spectrum(self, time_series: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def make_timeseries(self, spectrogram: np.ndarray) -> np.ndarray:
        pass

class FFT(FrequencySpectrumCalculator):
    def __init__(self, fs=44100, nfft=None, scaling='spectrum'):
        self.__fs = fs
        self.__nfft = nfft
        self.__scaling = scaling
        self.__freq = None

    def get_fs(self):
        return self.__fs

    def set_fs(self, fs: int):
        self.__fs = fs

    def get_freq(self):
        return self.__freq

    def make_spectrum(self, time_series: np.ndarray):
        yf = fft(time_series, n=self.__nfft)
        self.__freq = fftfreq(len(time_series), 1 / self.__fs)
        return yf

    def make_timeseries(self, spectrum: np.ndarray):
        return ifft(spectrum, n=self.__nfft)

class RFFT(FrequencySpectrumCalculator):
    def __init__(self, fs=44100, nfft=None, scaling='spectrum'):
        self.__fs = fs
        self.__nfft = nfft
        self.__scaling = scaling
        self.__freq = None

    def get_fs(self):
        return self.__fs

    def set_fs(self, fs):
        self.__fs = fs

    def get_freq(self):
        return self.__freq

    def make_spectrum(self, time_series: np.ndarray):
        if np.max(np.abs(time_series)) == 0:  # FIXED - Handle zero signal
            time_series_normalized = time_series
        else:
            time_series_normalized = time_series / np.max(np.abs(time_series))
        yf = rfft(time_series_normalized, n=self.__nfft)[1:]
        self.__freq = rfftfreq(len(time_series), 1 / self.__fs)[1:]
        if np.max(np.abs(yf)) == 0:  # FIXED - Handle zero spectrum
            yf_norm = yf
        else:
            yf_norm = yf / np.max(np.abs(yf))
        return yf_norm

    def make_timeseries(self, spectrum: np.ndarray):
        return irfft(spectrum, n=self.__nfft)

class STFT(FrequencySpectrumCalculator):
    def __init__(self, fs=44100, window='hann', nperseg=None, noverlap=None, nfft=None, scaling='spectrum'):
        self.__fs = fs
        self.__window = window
        self.__nperseg = nperseg
        self.__noverlap = noverlap
        self.__nfft = nfft
        self.__scaling = scaling
        self.__time = None
        self.__freq = None

    def get_time(self):
        return self.__time

    def get_freq(self):
        return self.__freq

    def make_spectrum(self, time_series: np.ndarray):
        f, t, Zxx = stft(time_series, fs=self.__fs, window=self.__window, nperseg=self.__nperseg,
                          noverlap=self.__noverlap, nfft=self.__nfft, scaling=self.__scaling)
        self.__time = t
        self.__freq = f
        return Zxx

    def make_timeseries(self, spectrogram: np.ndarray):
        time_series, _ = istft(spectrogram, fs=self.__fs, window=self.__window, nperseg=self.__nperseg,
                               noverlap=self.__noverlap, nfft=self.__nfft, scaling=self.__scaling)
        return time_series

# Communication Interface
class CommunicationInterface(ABC):  # FIXED - Now properly inherits from ABC
    @abstractmethod
    def read_sample(self) -> bytes:
        pass

    @abstractmethod
    def write_sample(self, sample: bytes) -> None:
        pass

class AudioStream(CommunicationInterface):
    def __init__(self, fs: int = 44100, buffer_seconds: float = 1, channels: int = 1, audio_format=pyaudio.paInt16, audio_input: bool = True, audio_output: bool = False):
        self.__p = pyaudio.PyAudio()
        self.__fs = fs
        self.__buffer_length = int(buffer_seconds * fs)
        self.__channels = channels
        self.__audio_format = audio_format
        self.__audio_input = audio_input
        self.__audio_output = audio_output
        self.__stream = None

    def start(self):
        self.__stream = self.__p.open(
            format=self.__audio_format,
            channels=self.__channels,
            rate=self.__fs,
            input=self.__audio_input,
            output=self.__audio_output,
            frames_per_buffer=self.__buffer_length
        )

    def read_sample(self) -> Datum:
        dtype = np.int16
        audio = np.frombuffer(self.__stream.read(self.__buffer_length), dtype=dtype)
        datum = Datum(DatumType.AUDIO)
        datum.add_raw_datum(RawDatumKey.AUDIO_ARRAY, audio)
        datum.add_raw_datum(RawDatumKey.SAMPLE_RATE, self.__fs)
        return datum

    def write_sample(self, sample: bytes) -> None:
        pass

    def close(self):
        if self.__stream:
            self.__stream.stop_stream()
            self.__stream.close()
        self.__p.terminate()

# Feature Extraction Classes
class FeatureExtraction(ABC):  # FIXED - Now properly inherits from ABC
    @abstractmethod
    def extract_features(self, signal: Datum) -> Datum:
        pass

class Decimation(FeatureExtraction):
    def __init__(self, target_rate: int = 500) -> None:
        self.__filter = Resample(new_rate=target_rate)

    def extract_features(self, signal: Datum) -> Datum:
        try:
            filtered_signal = self.__filter.apply_filter(
                signal.get_raw_datum(RawDatumKey.AUDIO_ARRAY),
                signal.get_raw_datum(RawDatumKey.SAMPLE_RATE)
            )
            signal.add_derived_data(DerivedDataKey.DECIMATED_AUDIO, filtered_signal)
            signal.add_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, self.__filter.get_new_rate())
        except Exception as e:
            print(f"✗ Decimation error: {e}")
            logging.error(f"Decimation error: {e}")
            signal.add_derived_data(DerivedDataKey.DECIMATED_AUDIO, signal.get_raw_datum(RawDatumKey.AUDIO_ARRAY))
            signal.add_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, signal.get_raw_datum(RawDatumKey.SAMPLE_RATE))
        return signal

class FrequencyPeakFinder(FeatureExtraction):
    def __init__(self, buffer_seconds: float = 10) -> None:
        self.__filter = TunableFilter(init_freq=100, bandwidth=10, order=4)
        self.__buffer = np.array([])
        self.__buffer_seconds = buffer_seconds

    def extract_features(self, signal: Datum) -> Datum:
        try:
            if DerivedDataKey.DECIMATED_AUDIO in signal.get_derived_data_keys():
                signal_array = signal.get_derived_data(DerivedDataKey.DECIMATED_AUDIO)
                sample_rate = signal.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE)
            else:
                signal_array = signal.get_raw_datum(RawDatumKey.AUDIO_ARRAY)
                sample_rate = signal.get_raw_datum(RawDatumKey.SAMPLE_RATE)
            self.__buffer = np.concatenate((self.__buffer, signal_array), axis=None)
            self.__buffer = self.__buffer[-int(sample_rate * self.__buffer_seconds):]
            filtered_signal = self.__filter.apply_filter(self.__buffer, sample_rate)
            signal.add_derived_data(DerivedDataKey.FREQUENCY_PEAK_FILTERED_AUDIO, filtered_signal)
            signal.add_derived_data(DerivedDataKey.FREQUENCY_PEAK, self.__filter.get_peak_freq())
        except Exception as e:
            print(f"✗ Peak finding error: {e}")
            logging.error(f"Peak finding error: {e}")
            signal.add_derived_data(DerivedDataKey.FREQUENCY_PEAK, 0.0)
            signal.add_derived_data(DerivedDataKey.FREQUENCY_PEAK_FILTERED_AUDIO, signal.get_raw_datum(RawDatumKey.AUDIO_ARRAY))
        return signal

class RPM(FeatureExtraction):
    def __init__(self, events_per_crankshaft_cycle: int) -> None:
        self.__events_per_crankshaft_cycle = events_per_crankshaft_cycle

    def extract_features(self, signal: Datum) -> Datum:
        try:
            peak = signal.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
            rpm = peak * 60 / self.__events_per_crankshaft_cycle
            if rpm < 0 or rpm > 10000:
                rpm = 0.0
            signal.add_derived_data(DerivedDataKey.RPM, rpm)
            engine_state = 1 if rpm > 100 else 0
            signal.add_derived_data(DerivedDataKey.ENGINE_STATE, engine_state)
        except Exception as e:
            print(f"✗ RPM calculation error: {e}")
            logging.error(f"RPM calculation error: {e}")
            signal.add_derived_data(DerivedDataKey.RPM, 0.0)
            signal.add_derived_data(DerivedDataKey.ENGINE_STATE, 0)
        return signal

# Feature Engineering Pipeline
class FeatureEngineeringPipeline:
    def __init__(self):
        self.__feature_engineering_blocks = []
        self.__output_blocks = []

    def add_block(self, block: FeatureExtraction) -> None:
        self.__feature_engineering_blocks.append(block)

    def add_output_block(self, block: FeatureExtraction) -> None:
        self.__output_blocks.append(block)

    def run(self, datum: Datum) -> Datum:
        for block in self.__feature_engineering_blocks:
            datum = block.extract_features(datum)
        for block in self.__output_blocks:
            datum = block.extract_features(datum)
        return datum

# UART and Disk Management Functions
def get_timestamp():
    return int(time.time() * 1000)

def calculate_crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc = crc << 1
            crc &= 0xFF
    return crc

def create_uart_packet(timestamp, datum: Datum, seq_num):
    try:
        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
        engine_state = datum.get_derived_data(DerivedDataKey.ENGINE_STATE) or 0
        peak_freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
        payload = bytearray()
        payload.extend(struct.pack('<H', seq_num))  # 2-byte sequence number
        payload.extend(struct.pack('<Q', timestamp))  # 8-byte timestamp (uint64_t ms)
        payload.extend(struct.pack('<f', float(rpm)))  # 4-byte RPM
        payload.extend(struct.pack('<B', int(engine_state)))  # 1-byte status
        payload.extend(struct.pack('<f', float(peak_freq)))  # 4-byte frequency
        length = len(payload)
        crc = calculate_crc8(payload)
        packet = bytearray([START_BYTE_DATA, length])
        packet.extend(payload)
        packet.append(crc)
        packet.append(END_BYTE)
        return packet
    except Exception as e:
        print(f"Packet creation error: {e}")
        logging.error(f"Packet creation error: {e}")
        return None

def send_uart_packet(timestamp, datum: Datum, seq_num):
    global packet_seq_num
    if ser is None or not ser.is_open:
        print("UART disconnected. Attempting to reconnect...")
        logging.warning("UART disconnected. Attempting to reconnect.")
        init_uart()
        if ser is None:
            return False
    with uart_lock:
        for attempt in range(MAX_RETRIES):
            try:
                packet = create_uart_packet(timestamp, datum, seq_num)
                if packet is None:
                    return False
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                bytes_written = ser.write(packet)
                ser.flush()
                if bytes_written != len(packet):
                    print(f"✗ Incomplete write: {bytes_written}/{len(packet)} bytes")
                    continue
                ser.timeout = ACK_TIMEOUT
                response = ser.read(3)
                if len(response) == 3 and response[0] == 0x06:
                    ack_seq = struct.unpack('<H', response[1:3])[0]
                    if ack_seq == seq_num:
                        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
                        status = datum.get_derived_data(DerivedDataKey.ENGINE_STATE) or 0
                        freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
                        print(f"✓ UART OK - Seq:{seq_num}, RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
                        logging.info(f"Packet sent - Seq:{seq_num}, RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
                        return True
                    else:
                        print(f"✗ ACK sequence mismatch: expected {seq_num}, got {ack_seq}")
                        logging.warning(f"ACK sequence mismatch: expected {seq_num}, got {ack_seq}")
                else:
                    print(f"✗ No/invalid ACK (attempt {attempt + 1}/{MAX_RETRIES})")
                    logging.warning(f"No/invalid ACK received on attempt {attempt + 1}")
            except Exception as e:
                print(f"✗ UART send error (attempt {attempt + 1}): {e}")
                logging.error(f"UART send error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INTER_PACKET_DELAY * (attempt + 1))
        print(f"✗ Failed to send packet after {MAX_RETRIES} attempts")
        logging.error(f"Failed to send packet after {MAX_RETRIES} attempts")
        return False

def send_record_alert(timestamp):
    if ser is None:
        return
    with uart_lock:
        try:
            payload = struct.pack('<Q', timestamp)
            crc = calculate_crc8(payload)
            packet = bytearray([START_BYTE_RECORD, len(payload)])
            packet.extend(payload)
            packet.append(crc)
            packet.append(END_BYTE)
            ser.write(packet)
            ser.flush()
            print(f"Record alert sent: {timestamp}")
            logging.info(f"Record alert sent: {timestamp}")
        except Exception as e:
            print(f"✗ Record alert error: {e}")
            logging.error(f"Record alert error: {e}")

def send_sync_packet():
    if ser is None:
        return
    with uart_lock:
        try:
            current_time = get_timestamp()
            payload = struct.pack('<Q', current_time)
            crc = calculate_crc8(payload)
            packet = bytearray([START_BYTE_SYNC, len(payload)])
            packet.extend(payload)
            packet.append(crc)
            packet.append(END_BYTE)
            ser.write(packet)
            ser.flush()
            print(f"Time sync sent: {current_time}")
            logging.info(f"Time sync sent: {current_time}")
        except Exception as e:
            print(f"✗ Sync error: {e}")
            logging.error(f"Sync error: {e}")

def check_disk_space():
    try:
        disk = psutil.disk_usage(AUDIO_DIR)
        free_space = disk.free / (1024 ** 3)
        free_percent = disk.free / disk.total
        return free_space < MIN_FREE_SPACE_GB or free_percent < DISK_SPACE_THRESHOLD
    except Exception as e:
        print(f"✗ Disk space check error: {e}")
        return False

def cleanup_old_files():
    print("🧹 Cleaning old recordings...")
    logging.info("Cleaning old recordings")
    deleted_count = 0
    try:
        files = [(f, os.path.getmtime(os.path.join(AUDIO_DIR, f)))
                 for f in os.listdir(AUDIO_DIR)
                 if os.path.isfile(os.path.join(AUDIO_DIR, f)) and f.endswith('.wav')]
        files.sort(key=lambda x: x[1])
        for file, _ in files:
            if not check_disk_space():
                break
            path = os.path.join(AUDIO_DIR, file)
            try:
                os.remove(path)
                deleted_count += 1
                print(f"Deleted: {file}")
                logging.info(f"Deleted: {file}")
            except Exception as e:
                print(f"✗ Delete error: {e}")
                logging.error(f"Delete error: {e}")
    except Exception as e:
        print(f"✗ Cleanup error: {e}")
        logging.error(f"Cleanup error: {e}")
    print(f"✓ Cleanup complete. Deleted {deleted_count} files.")
    logging.info(f"Cleanup complete. Deleted {deleted_count} files.")
    return deleted_count > 0

# Recorder and Analyzer
def recorder():
    global file_counter, last_cleanup, packet_seq_num
    print(f"Recording in {AUDIO_MODE} mode")
    logging.info(f"Recording in {AUDIO_MODE} mode")

    audio_stream = None
    if AUDIO_MODE == "stream":
        audio_stream = AudioStream(fs=SAMPLE_RATE, buffer_seconds=AUDIO_BUFFER_SECONDS, channels=1)
        audio_stream.start()

    while True:
        try:
            if check_disk_space():
                cleanup_old_files()
            if analyze_queue.full():
                print(f"Analyze queue full ({analyze_queue.qsize()}/{ANALYZE_QUEUE_SIZE}). Dropping oldest item.")
                logging.warning(f"Analyze queue full. Dropping oldest item.")
                try:
                    analyze_queue.get_nowait()
                except queue.Empty:
                    pass
            record_start_time = get_timestamp()
            send_record_alert(record_start_time)

            if AUDIO_MODE == "file":
                filename = f"Rec{file_counter:06d}.wav"
                filepath = os.path.join(AUDIO_DIR, filename)
                print(f"\n Recording: {filename}")
                logging.info(f"Recording: {filename}")
                audio = sd.rec(
                    int(RECORD_SECONDS * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='int16',
                    device=MIC_DEVICE
                )
                sd.wait()
                wav.write(filepath, SAMPLE_RATE, audio.flatten())
                print(f"Saved: {filename}")
                logging.info(f"Saved: {filename}")
                analyze_queue.put((filepath, record_start_time, packet_seq_num))
                file_counter += 1
            else:  # AUDIO_MODE == "stream"
                datum = audio_stream.read_sample()
                analyze_queue.put((datum, record_start_time, packet_seq_num))

            packet_seq_num = (packet_seq_num + 1) % 65536
            if time.time() - last_cleanup > CLEAN_INTERVAL_DAYS * 86400:
                threading.Thread(target=cleanup_old_files, daemon=True).start()
                last_cleanup = time.time()
        except Exception as e:
            print(f"✗ Recording error: {e}")
            logging.error(f"Recording error: {e}")
            time.sleep(1)

    if audio_stream:
        audio_stream.close()

def analyzer():
    pipeline = FeatureEngineeringPipeline()
    pipeline.add_block(Decimation(target_rate=DECIMATED_RATE))
    pipeline.add_block(FrequencyPeakFinder(buffer_seconds=FFT_BUFFER_SECONDS))
    pipeline.add_output_block(RPM(events_per_crankshaft_cycle=EVENTS_PER_CYCLE))
    while True:
        try:
            item = analyze_queue.get()
            if AUDIO_MODE == "file":
                filepath, record_timestamp, seq_num = item
                print(f"Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
                logging.info(f"Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
                sample_rate, data = wav.read(filepath)
                if len(data.shape) > 1:
                    data = data.flatten()
                datum = Datum(DatumType.AUDIO, audio_array=data, sample_rate=sample_rate)
            else:  # AUDIO_MODE == "stream"
                datum, record_timestamp, seq_num = item
                print(f"Analyzing: Streamed audio (Queue: {analyze_queue.qsize()})")
                logging.info(f"Analyzing: Streamed audio (Queue: {analyze_queue.qsize()})")

            datum = pipeline.run(datum)
            success = send_uart_packet(record_timestamp, datum, seq_num)
            if success:
                rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
                status = datum.get_derived_data(DerivedDataKey.ENGINE_STATE) or 0
                freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
                print(f"   ⚡ RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
        except Exception as e:
            print(f"Analysis error: {e}")
            logging.error(f"Analysis error: {e}")

# Main Function
def main():
    print("Audio Processing System Starting...")
    logging.info("Audio Processing System Starting")
    print(f"Recordings directory: {AUDIO_DIR}")
    init_uart()  # FIXED - Removed the condition check
    send_sync_packet()
    recorder_thread = threading.Thread(target=recorder, daemon=True)
    analyzer_thread = threading.Thread(target=analyzer, daemon=True)
    recorder_thread.start()
    analyzer_thread.start()
    print("System started!")
    logging.info("System started")
    print("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(300)
            send_sync_packet()
    except KeyboardInterrupt:
        print("\n Stopping system...")
        logging.info("Stopping system")
        if ser:
            ser.close()
        print("System stopped")
        logging.info("System stopped")

if __name__ == "__main__":
    main()