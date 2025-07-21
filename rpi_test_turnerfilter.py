import os
import time
import threading
import queue
import struct
import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fftpack
from scipy.signal import decimate, butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import glob
import logging
import csv
from datetime import datetime
from typing import Tuple

# Settings
AUDIO_DIR =  "./recordings"  # "./BANJI_WORK_CODE/simulation_data/audios" #
UART_PORT = "/dev/serial0"  # Change to your ESP32 port (e.g., "/dev/ttyUSB0")
UART_BAUDRATE = 115200
DECIMATED_RATE = 500
EVENTS_PER_CYCLE = 2
ANALYZE_QUEUE_SIZE = 10
LOG_FILE = "audio_system_test.log"

# CSV Results Settings
RESULTS_CSV_FILE = "audio_analysis_results.csv"
CSV_HEADERS = [
    'timestamp',
    'datetime',
    'filename',
    'sequence_number',
    'sample_rate',
    'audio_duration_seconds',
    'decimated_rate',
    'frequency_peak_hz',
    'rpm',
    'engine_status',
    'processing_time_ms',
    'buffer_size_samples',
    'peak_amplitude',
    'notes'
]

# Test mode settings
TEST_MODE = True
PROCESS_INTERVAL = 2.0  # Seconds between processing files (simulates recording interval)
LOOP_PROCESSING = True  # Set to True to continuously loop through files

# UART Protocol Constants (keep for compatibility)
START_BYTE_DATA = 0xAA
START_BYTE_RECORD = 0xAC
START_BYTE_SYNC = 0xAB
END_BYTE = 0x55
ACK_TIMEOUT = 4.0
MAX_RETRIES = 2
INTER_PACKET_DELAY = 0.05

# Setup logging
logging.basicConfig(
    filename=LOG_FILE, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Changed to INFO for test mode
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

os.makedirs(AUDIO_DIR, exist_ok=True)

# Queues
analyze_queue = queue.Queue(maxsize=ANALYZE_QUEUE_SIZE)

# CSV file lock for thread safety
csv_lock = threading.Lock()

def initialize_csv_file():
    """Initialize the CSV file with headers if it doesn't exist"""
    if not os.path.exists(RESULTS_CSV_FILE):
        with open(RESULTS_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADERS)
        print(f"📊 Created CSV results file: {RESULTS_CSV_FILE}")
        logging.info(f"Created CSV results file: {RESULTS_CSV_FILE}")
    else:
        print(f"📊 Using existing CSV results file: {RESULTS_CSV_FILE}")

def save_results_to_csv(analysis_data):
    """Save analysis results to CSV file in a thread-safe manner"""
    try:
        with csv_lock:
            with open(RESULTS_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Prepare the row data
                row = [
                    analysis_data.get('timestamp', ''),
                    analysis_data.get('datetime', ''),
                    analysis_data.get('filename', ''),
                    analysis_data.get('sequence_number', ''),
                    analysis_data.get('sample_rate', ''),
                    analysis_data.get('audio_duration_seconds', ''),
                    analysis_data.get('decimated_rate', ''),
                    analysis_data.get('frequency_peak_hz', ''),
                    analysis_data.get('rpm', ''),
                    analysis_data.get('engine_status', ''),
                    analysis_data.get('processing_time_ms', ''),
                    analysis_data.get('buffer_size_samples', ''),
                    analysis_data.get('peak_amplitude', ''),
                    analysis_data.get('notes', '')
                ]
                
                writer.writerow(row)
                
        print(f"💾 Results saved to CSV: {analysis_data.get('filename', 'unknown')}")
        logging.info(f"Results saved to CSV for file: {analysis_data.get('filename', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")
        logging.error(f"Error saving to CSV: {e}")

# Mock UART for testing
class MockUART:
    def __init__(self):
        self.is_open = True
        self.packet_count = 0
        
    def send_packet(self, timestamp, datum, seq_num, analysis_data=None):
        """Mock UART send function that just logs the data"""
        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
        status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0
        freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
        
        print(f"✓ MOCK UART - Seq:{seq_num}, RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
        logging.info(f"Mock packet sent - Seq:{seq_num}, RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
        self.packet_count += 1
        
        # Save to CSV if analysis data is provided
        if analysis_data:
            save_results_to_csv(analysis_data)
            
        return True
        
    def send_record_alert(self, timestamp):
        print(f"📡 MOCK: Record alert sent: {timestamp}")
        
    def send_sync_packet(self):
        print(f"📡 MOCK: Sync packet sent: {int(time.time() * 1000)}")

# Global mock UART instance
mock_uart = MockUART()

# Datum classes (using original approach)
class RawDatumKey:
    AUDIO_ARRAY = "audio_array"
    SAMPLE_RATE = "sample_rate"

class DerivedDataKey:
    DECIMATED_AUDIO = "decimated_audio"
    DECIMATED_AUDIO_RATE = "decimated_audio_rate"
    FREQUENCY_PEAK = "frequency_peak"
    FREQUENCY_PEAK_FILTERED_AUDIO = "frequency_peak_filtered_audio"
    RPM = "rpm"
    ENGINE_STATUS = "engine_status"
    PEAK_AMPLITUDE = "peak_amplitude"  # New field for CSV
    BUFFER_SIZE = "buffer_size"  # New field for CSV

class Datum:
    def __init__(self, audio_array, sample_rate):
        self._raw_data = {RawDatumKey.AUDIO_ARRAY: audio_array, RawDatumKey.SAMPLE_RATE: sample_rate}
        self._derived_data = {}

    def get_raw_datum(self, key):
        return self._raw_data.get(key)

    def set_raw_datum(self, key, value):
        self._raw_data[key] = value

    def get_derived_data(self, key):
        return self._derived_data.get(key)

    def set_derived_data(self, key, value):
        self._derived_data[key] = value
    
    def get_derived_data_keys(self):
        return list(self._derived_data.keys())

    def add_derived_data(self, key, value):
        self._derived_data[key] = value

    def add_raw_datum(self, key, value):
        self._raw_data[key] = value

# FeatureExtraction interface (unchanged)
class FeatureExtraction:
    def extract_features(self, datum: Datum) -> Datum:
        raise NotImplementedError

# ORIGINAL APPROACH: BandPassFilter class from your first code
class BandPassFilter:
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

# ORIGINAL APPROACH: FFT class from your first code
class FFT:
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

# ORIGINAL APPROACH: TunableFilter class from your first code
class TunableFilter:
    def __init__(self, init_freq: float = 125, bandwidth: float = 50, order: int = 2) -> None:
        self.__peak_freq = init_freq
        self.__bandwidth = bandwidth
        self.__order = order
        self.__peak_amplitude = 0.0  # Store peak amplitude for CSV
        
        self.__bp_filter = BandPassFilter(
            lowcut=10,
            highcut=200,
            order=self.__order,
            analog=False
        )
        
        self.__fft = None

    def get_peak_freq(self) -> float:
        return self.__peak_freq

    def get_peak_amplitude(self) -> float:
        return self.__peak_amplitude

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

        # Find the peaks in the power spectrum - IMPROVED APPROACH
        peak_indices = find_peaks(power_spectrum, distance=sample_rate//10)[0]  # Fixed: better distance parameter
        peak_frequencies = self.__fft.get_freq()[peak_indices]

        # If there are no peaks, return the filtered signal
        if len(peak_indices) == 0:
            self.__peak_amplitude = 0.0
            return filtered_signal

        # FIXED: Find the peak with highest amplitude (not minimum frequency)
        # This is crucial for engine RPM detection - we want the fundamental frequency
        peak_powers = power_spectrum[peak_indices]
        max_power_idx = np.argmax(peak_powers)
        peak_index = peak_indices[max_power_idx]

        # Store the peak amplitude for CSV logging
        self.__peak_amplitude = peak_powers[max_power_idx]

        # Find the frequency of the highest amplitude peak
        peak_frequency = self.__fft.get_freq()[peak_index]

        # Set the peak frequency
        self.set_peak_freq(abs(peak_frequency))  # Fixed: ensure positive frequency

        # Return the filtered signal
        return filtered_signal

# ORIGINAL APPROACH: Resample class from your first code
class Resample:
    def __init__(self, new_rate: int) -> None:
        self.__new_rate = new_rate

    def get_new_rate(self) -> int:
        return self.__new_rate

    def set_new_rate(self, new_rate: int):
        self.__new_rate = new_rate

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        from scipy.signal import resample
        resampled_length = int(len(signal) * self.__new_rate / sample_rate)
        return resample(signal, resampled_length)

# Feature engineering blocks - ORIGINAL APPROACH
class Decimation(FeatureExtraction):
    def __init__(self, target_rate: int = 500) -> None:
        self.__filter = Resample(new_rate=target_rate)

    def extract_features(self, signal: Datum) -> Datum:
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
        self.__filter = TunableFilter(init_freq=100, bandwidth=10, order=4)
        # Initialize the buffer to an empty array
        self.__buffer = np.array([])
        # Store the buffer size in seconds
        self.__buffer_seconds = buffer_seconds

    def extract_features(self, signal: Datum) -> Datum:
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

        # Add the frequency peak to the signal's derived data - ORIGINAL APPROACH
        signal.add_derived_data(DerivedDataKey.FREQUENCY_PEAK, self.__filter.get_peak_freq())
        
        # Add peak amplitude and buffer size for CSV logging
        signal.add_derived_data(DerivedDataKey.PEAK_AMPLITUDE, self.__filter.get_peak_amplitude())
        signal.add_derived_data(DerivedDataKey.BUFFER_SIZE, len(self.__buffer))

        return signal

class RPM(FeatureExtraction):
    def __init__(self, events_per_crankshaft_cycle: int) -> None:
        self.__events_per_crankshaft_cycle = events_per_crankshaft_cycle

    def extract_features(self, signal: Datum) -> Datum:
        # Get the frequency peak of the signal - ORIGINAL APPROACH
        peak = signal.get_derived_data(DerivedDataKey.FREQUENCY_PEAK)

        # Calculate the RPM based on the frequency peak - ORIGINAL APPROACH
        rpm = self.__calculate_rpm(peak)

        # Add the RPM to the signal's derived data
        signal.add_derived_data(DerivedDataKey.RPM, rpm)
        
        # Add engine status
        engine_status = 1 if rpm and rpm > 100 else 0  # Fixed: handle None values
        signal.add_derived_data(DerivedDataKey.ENGINE_STATUS, engine_status)

        return signal

    def __calculate_rpm(self, peak: float) -> float:
        # ORIGINAL APPROACH: RPM calculation
        if peak is None or peak <= 0:  # Fixed: handle None and zero values
            return 0.0
        return peak * 60 / self.__events_per_crankshaft_cycle

# FeatureEngineeringPipeline (unchanged)
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

def get_timestamp():
    return int(time.time() * 1000)

def get_audio_files():
    """Get all WAV files from the recordings directory"""
    pattern = os.path.join(AUDIO_DIR, "*.wav")
    files = glob.glob(pattern)
    files.sort()  # Process in alphabetical order
    return files

def file_processor():
    """Process existing audio files instead of recording new ones"""
    global analyze_queue
    packet_seq_num = 0
    
    audio_files = get_audio_files()
    if not audio_files:
        print(f"❌ No WAV files found in {AUDIO_DIR}")
        logging.error(f"No WAV files found in {AUDIO_DIR}")
        return
    
    print(f"📁 Found {len(audio_files)} audio files to process")
    logging.info(f"Found {len(audio_files)} audio files to process")
    
    file_index = 0
    
    while True:
        try:
            # Handle queue overflow by dropping oldest item
            if analyze_queue.full():
                print(f"Analyze queue full ({analyze_queue.qsize()}/{ANALYZE_QUEUE_SIZE}). Dropping oldest item.")
                logging.warning(f"Analyze queue full. Dropping oldest item.")
                try:
                    analyze_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Get current file
            current_file = audio_files[file_index]
            process_start_time = get_timestamp()
            
            # Send mock record alert
            mock_uart.send_record_alert(process_start_time)
            
            print(f"\n🎵 Processing: {os.path.basename(current_file)} ({file_index + 1}/{len(audio_files)})")
            logging.info(f"Processing: {os.path.basename(current_file)}")
            
            # Add to analysis queue
            analyze_queue.put((current_file, process_start_time, packet_seq_num))
            
            packet_seq_num = (packet_seq_num + 1) % 65536
            file_index += 1
            
            # Check if we should loop or stop
            if file_index >= len(audio_files):
                if LOOP_PROCESSING:
                    print(f"🔄 Looping back to start (processed all {len(audio_files)} files)")
                    file_index = 0
                else:
                    print(f"✅ Finished processing all {len(audio_files)} files")
                    break
            
            # Wait before processing next file
            time.sleep(PROCESS_INTERVAL)
            
        except Exception as e:
            print(f"✗ File processing error: {e}")
            logging.error(f"File processing error: {e}")
            time.sleep(1)

def analyzer():
    """Analyze audio files using ORIGINAL APPROACH"""
    pipeline = FeatureEngineeringPipeline()
    pipeline.add_block(Decimation(target_rate=DECIMATED_RATE))
    pipeline.add_block(FrequencyPeakFinder(buffer_seconds=10))  # Use original 10 seconds
    pipeline.add_output_block(RPM(events_per_crankshaft_cycle=EVENTS_PER_CYCLE))
    
    while True:
        try:
            filepath, process_timestamp, seq_num = analyze_queue.get()
            analysis_start_time = time.time()
            
            print(f"🔬 Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
            logging.info(f"Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
            
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"❌ File not found: {filepath}")
                continue
            
            sample_rate, data = wav.read(filepath)
            if len(data.shape) > 1:
                data = data.flatten()
                
            # Calculate audio duration
            audio_duration = len(data) / sample_rate
                
            datum = Datum(audio_array=data, sample_rate=sample_rate)
            datum = pipeline.run(datum)
            
            # Calculate processing time
            processing_time_ms = (time.time() - analysis_start_time) * 1000
            
            # Prepare analysis data for CSV
            analysis_data = {
                'timestamp': process_timestamp,
                'datetime': datetime.fromtimestamp(process_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'filename': os.path.basename(filepath),
                'sequence_number': seq_num,
                'sample_rate': sample_rate,
                'audio_duration_seconds': round(audio_duration, 3),
                'decimated_rate': datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE) or 0,
                'frequency_peak_hz': round(datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0, 2),
                'rpm': round(datum.get_derived_data(DerivedDataKey.RPM) or 0.0, 1),
                'engine_status': datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0,
                'processing_time_ms': round(processing_time_ms, 2),
                'buffer_size_samples': datum.get_derived_data(DerivedDataKey.BUFFER_SIZE) or 0,
                'peak_amplitude': round(datum.get_derived_data(DerivedDataKey.PEAK_AMPLITUDE) or 0.0, 6),
                'notes': f"Events per cycle: {EVENTS_PER_CYCLE}"
            }
            
            # Use mock UART instead of real UART, passing analysis data for CSV saving
            success = mock_uart.send_packet(process_timestamp, datum, seq_num, analysis_data)
            
            if success:
                rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
                status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0
                freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
                print(f"   ⚡ Analysis complete: RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz, Duration:{audio_duration:.1f}s")
            else:
                print(f"   ❌ Failed to send analysis results")
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"✗ Analysis error: {e}")
            logging.error(f"Analysis error: {e}")

def status_monitor():
    """Background status monitoring"""
    while True:
        try:
            time.sleep(60)  # Every minute
            
            # Count CSV entries
            csv_entries = 0
            try:
                if os.path.exists(RESULTS_CSV_FILE):
                    with open(RESULTS_CSV_FILE, 'r') as f:
                        csv_entries = sum(1 for line in f) - 1  # Subtract header row
            except:
                csv_entries = 0
            
            print(f"📊 Status: Queue: {analyze_queue.qsize()}/{ANALYZE_QUEUE_SIZE}, "
                  f"Packets sent: {mock_uart.packet_count}, CSV entries: {csv_entries}")
                      
            logging.info(f"System status - Queue: {analyze_queue.qsize()}, "
                        f"Packets sent: {mock_uart.packet_count}, CSV entries: {csv_entries}")
                        
        except Exception as e:
            logging.error(f"Status monitor error: {e}")

def main():
    print("🧪 Audio Processing System - TEST MODE (IMPROVED ORIGINAL APPROACH + CSV)")
    print(f"📁 Processing files from: {AUDIO_DIR}")
    print(f"📊 Results CSV file: {RESULTS_CSV_FILE}")
    print(f"⏱️ Process interval: {PROCESS_INTERVAL} seconds")
    print(f"🔄 Loop processing: {LOOP_PROCESSING}")
    print(f"🎯 Target decimated rate: {DECIMATED_RATE} Hz")
    print(f"🚗 Events per cycle: {EVENTS_PER_CYCLE}")
    print(f"🔧 APPROACH: TunableFilter with adaptive bandpass, HIGHEST AMPLITUDE peak detection, rolling buffer")
    
    logging.info("Audio Processing System - TEST MODE (IMPROVED ORIGINAL APPROACH + CSV) Starting")
    
    # Initialize CSV file
    initialize_csv_file()
    
    # Check if recordings directory exists and has files
    audio_files = get_audio_files()
    if not audio_files:
        print(f"❌ No WAV files found in {AUDIO_DIR}")
        print("   Please ensure you have WAV files in the recordings directory")
        return
    
    print(f"✅ Found {len(audio_files)} WAV files to process")
    
    # Send initial mock sync packet
    mock_uart.send_sync_packet()
    
    # Start all threads
    threads = []
    
    # Status monitoring thread
    status_thread = threading.Thread(target=status_monitor, daemon=True)
    status_thread.start()
    threads.append(status_thread)
    
    # Main processing threads
    processor_thread = threading.Thread(target=file_processor, daemon=True)
    analyzer_thread = threading.Thread(target=analyzer, daemon=True)
    
    processor_thread.start()
    analyzer_thread.start()
    threads.extend([processor_thread, analyzer_thread])
    
    print("✅ Improved original approach test system started successfully!")
    print("📊 Active threads:")
    print("   🎵 File processor (replacing recorder)")
    print("   🔬 Signal analyzer (IMPROVED: TunableFilter + HIGHEST AMPLITUDE peak detection)") 
    print("   📈 Status monitor")
    print("   💾 CSV results logging enabled")
    print("\n💻 Commands:")
    print("   - Press 's' + Enter for status")
    print("   - Press 'q' + Enter to quit")
    print("   - Press 'csv' + Enter to show CSV info")
    print("   - Press Ctrl+C for emergency stop")
    
    logging.info("All threads started successfully")
    
    try:
        while True:
            # Handle user input
            try:
                user_input = input().strip().lower()
                if user_input == 'q':
                    print("User requested shutdown")
                    break
                elif user_input == 's':
                    # Count CSV entries
                    csv_entries = 0
                    try:
                        if os.path.exists(RESULTS_CSV_FILE):
                            with open(RESULTS_CSV_FILE, 'r') as f:
                                csv_entries = sum(1 for line in f) - 1  # Subtract header row
                    except:
                        csv_entries = 0
                    
                    print(f"\n📊 System Status:")
                    print(f"   Queue: {analyze_queue.qsize()}/{ANALYZE_QUEUE_SIZE}")
                    print(f"   Packets sent: {mock_uart.packet_count}")
                    print(f"   CSV entries: {csv_entries}")
                    print(f"   Files available: {len(get_audio_files())}")
                    print(f"   Loop mode: {LOOP_PROCESSING}")
                    print(f"   Process interval: {PROCESS_INTERVAL}s")
                    print(f"   CSV file: {RESULTS_CSV_FILE}")
                    print(f"   Approach: TunableFilter with adaptive bandpass, HIGHEST AMPLITUDE peak\n")
                elif user_input == 'csv':
                    # Show CSV file information
                    if os.path.exists(RESULTS_CSV_FILE):
                        try:
                            with open(RESULTS_CSV_FILE, 'r') as f:
                                lines = f.readlines()
                                total_entries = len(lines) - 1  # Subtract header
                                
                            print(f"\n📊 CSV File Information:")
                            print(f"   File: {RESULTS_CSV_FILE}")
                            print(f"   Total entries: {total_entries}")
                            print(f"   File size: {os.path.getsize(RESULTS_CSV_FILE)} bytes")
                            
                            if total_entries > 0:
                                print(f"   Headers: {', '.join(CSV_HEADERS)}")
                                if len(lines) > 1:
                                    print(f"   Last entry preview: {lines[-1].strip()[:100]}...")
                            print()
                        except Exception as e:
                            print(f"❌ Error reading CSV file: {e}")
                    else:
                        print(f"\n❌ CSV file {RESULTS_CSV_FILE} does not exist yet")
                elif user_input == 'sync':
                    mock_uart.send_sync_packet()
                    print("Mock sync packet sent")
            except EOFError:
                # Handle Ctrl+D
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C in input
                break
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received...")
        
    finally:
        print("🔄 Stopping improved original approach test system...")
        logging.info("Stopping improved original approach test system")
        
        # Final CSV statistics
        if os.path.exists(RESULTS_CSV_FILE):
            try:
                with open(RESULTS_CSV_FILE, 'r') as f:
                    total_entries = sum(1 for line in f) - 1  # Subtract header
                print(f"📊 Final CSV summary: {total_entries} analysis results saved to {RESULTS_CSV_FILE}")
                logging.info(f"Final CSV summary: {total_entries} analysis results saved")
            except:
                pass
        
        print("✅ Improved original approach test system stopped cleanly")
        logging.info("Improved original approach test system stopped")

if __name__ == "__main__":
    main()