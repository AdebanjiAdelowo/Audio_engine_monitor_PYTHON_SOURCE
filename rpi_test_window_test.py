import os
import time
import threading
import queue
import struct
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample, butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import glob
import logging

# Settings
AUDIO_DIR = "./recordings"
UART_PORT = "/dev/serial0"  # Change to your ESP32 port (e.g., "/dev/ttyUSB0")
UART_BAUDRATE = 115200
DECIMATED_RATE = 500
EVENTS_PER_CYCLE = 2
ANALYZE_QUEUE_SIZE = 10
LOG_FILE = "audio_system_test.log"

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

# Mock UART for testing
class MockUART:
    def __init__(self):
        self.is_open = True
        self.packet_count = 0
        
    def send_packet(self, timestamp, datum, seq_num):
        """Mock UART send function that just logs the data"""
        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
        status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0
        freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
        
        print(f"✓ MOCK UART - Seq:{seq_num}, RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
        logging.info(f"Mock packet sent - Seq:{seq_num}, RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
        self.packet_count += 1
        return True
        
    def send_record_alert(self, timestamp):
        print(f"📡 MOCK: Record alert sent: {timestamp}")
        
    def send_sync_packet(self):
        print(f"📡 MOCK: Sync packet sent: {int(time.time() * 1000)}")

# Global mock UART instance
mock_uart = MockUART()

# Datum classes (unchanged)
class RawDatumKey:
    AUDIO_ARRAY = "audio_array"
    SAMPLE_RATE = "sample_rate"

class DerivedDataKey:
    DECIMATED_AUDIO = "decimated_audio"
    DECIMATED_AUDIO_RATE = "decimated_audio_rate"
    FREQUENCY_PEAK = "frequency_peak"
    DECIMATED_FREQUENCY_PEAK = "decimated_frequency_peak"
    RPM = "rpm"
    ENGINE_STATUS = "engine_status"

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

# FeatureExtraction interface (unchanged)
class FeatureExtraction:
    def extract_features(self, datum: Datum) -> Datum:
        raise NotImplementedError

class BandpassFilter:
    """Bandpass filter for signal preprocessing."""
    
    def __init__(self, lowcut: float = 20, highcut: float = 150, order: int = 4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
    
    def apply(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply bandpass filter to signal."""
        nyquist = sample_rate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, signal)

# Feature engineering blocks (IMPROVED)
class Decimation(FeatureExtraction):
    def __init__(self, target_rate=500):
        self.target_rate = target_rate

    def extract_features(self, datum: Datum):
        signal = datum.get_raw_datum(RawDatumKey.AUDIO_ARRAY)
        original_rate = datum.get_raw_datum(RawDatumKey.SAMPLE_RATE)
        
        try:
            # Use scipy.signal.resample instead of decimate for more precise control
            new_length = int(len(signal) * self.target_rate / original_rate)
            decimated_signal = resample(signal, new_length)
            decimated_rate = self.target_rate
            
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO, decimated_signal)
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, decimated_rate)
            
        except Exception as e:
            print(f"✗ Decimation error: {e}")
            logging.error(f"Decimation error: {e}")
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO, signal)
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, original_rate)
        
        return datum

class FrequencyPeakFinder(FeatureExtraction):
    def __init__(self, buffer_seconds=5):
        self.buffer_seconds = buffer_seconds
        # Initialize bandpass filter (20-150 Hz range)
        self.bandpass_filter = BandpassFilter(lowcut=20, highcut=150, order=4)
        # Rolling buffer for better frequency analysis
        self._buffer = np.array([])

    def _find_peak_with_scipy_peaks(self, signal: np.ndarray, sample_rate: int) -> float:
        """Find peak frequency using scipy.signal.find_peaks for better peak detection."""
        try:
            # Remove DC component
            signal = signal - np.mean(signal)
            
            # Apply bandpass filter
            filtered_signal = self.bandpass_filter.apply(signal, sample_rate)
            
            # Apply FFT
            N = len(filtered_signal)
            fft_data = np.abs(fft(filtered_signal)[:N//2])
            freqs = fftfreq(N, 1/sample_rate)[:N//2]
            
            # Find peaks with proper distance parameter
            min_distance = max(1, sample_rate // 40)  # Minimum 25 Hz separation for 500 Hz sample rate
            peaks, _ = find_peaks(fft_data, distance=min_distance)
            
            if len(peaks) == 0:
                return 0.0
            
            # Get the peak with highest amplitude
            peak_idx = peaks[np.argmax(fft_data[peaks])]
            return freqs[peak_idx]
            
        except Exception as e:
            logging.error(f"Peak finding error: {e}")
            return 0.0

    def extract_features(self, datum: Datum):
        try:
            # Process decimated signal for efficiency (like your original approach)
            decimated_signal = datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO)
            decimated_rate = datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE)
            
            if decimated_signal is not None and len(decimated_signal) > 0:
                # Update rolling buffer for better frequency analysis
                self._buffer = np.concatenate([self._buffer, decimated_signal])
                max_buffer_size = int(decimated_rate * self.buffer_seconds)
                self._buffer = self._buffer[-max_buffer_size:]
                
                # Use improved peak finding with scipy peaks and bandpass filtering
                if len(self._buffer) >= decimated_rate:  # Ensure we have at least 1 second of data
                    peak_freq = self._find_peak_with_scipy_peaks(self._buffer, decimated_rate)
                else:
                    # Fallback to simple method for insufficient data
                    peak_freq = self._find_peak_with_scipy_peaks(decimated_signal, decimated_rate)
                
                datum.set_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK, peak_freq)
            else:
                datum.set_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK, 0.0)
            
            # Also process original signal for compatibility (optional)
            signal = datum.get_raw_datum(RawDatumKey.AUDIO_ARRAY)
            sample_rate = datum.get_raw_datum(RawDatumKey.SAMPLE_RATE)
            
            # Simple FFT on original signal (keeping your windowing approach)
            windowed_signal = signal * np.hanning(len(signal))
            N = len(windowed_signal)
            fft_data = np.abs(fft(windowed_signal)[:N//2])
            freqs = fftfreq(N, 1/sample_rate)[:N//2]
            
            # Use consistent frequency range (20-150 Hz)
            valid_range = (freqs >= 20) & (freqs <= 150)
            if np.any(valid_range):
                valid_fft = fft_data[valid_range]
                valid_freqs = freqs[valid_range]
                peak_idx = np.argmax(valid_fft)
                peak_freq = valid_freqs[peak_idx]
            else:
                peak_freq = 0.0
            
            datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, peak_freq)
                
        except Exception as e:
            print(f"✗ Peak finding error: {e}")
            logging.error(f"Peak finding error: {e}")
            datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, 0.0)
            datum.set_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK, 0.0)
        
        return datum

class RPM(FeatureExtraction):
    def __init__(self, events_per_crankshaft_cycle=2):
        self.events_per_cycle = events_per_crankshaft_cycle

    def extract_features(self, datum: Datum):
        try:
            # Use decimated frequency peak (more accurate due to better processing)
            peak_freq = datum.get_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK)
            if peak_freq is None or peak_freq <= 0:
                peak_freq = 0.0
            
            # Calculate RPM
            rpm = (peak_freq * 60) / self.events_per_cycle
            
            # Validate RPM range (same as your original)
            if rpm < 0 or rpm > 10000:
                rpm = 0.0
            
            datum.set_derived_data(DerivedDataKey.RPM, rpm)
            
            # Engine status with slightly higher threshold for stability
            engine_status = 1 if rpm > 150 else 0  # Increased from 100 to reduce false positives
            datum.set_derived_data(DerivedDataKey.ENGINE_STATUS, engine_status)
            
        except Exception as e:
            print(f"✗ RPM calculation error: {e}")
            logging.error(f"RPM calculation error: {e}")
            datum.set_derived_data(DerivedDataKey.RPM, 0.0)
            datum.set_derived_data(DerivedDataKey.ENGINE_STATUS, 0)
        
        return datum

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
    """Analyze audio files with IMPROVED processing pipeline"""
    pipeline = FeatureEngineeringPipeline()
    pipeline.add_block(Decimation(target_rate=DECIMATED_RATE))
    pipeline.add_block(FrequencyPeakFinder(buffer_seconds=5))  # Keep 5 seconds as default
    pipeline.add_output_block(RPM(events_per_crankshaft_cycle=EVENTS_PER_CYCLE))
    
    while True:
        try:
            filepath, process_timestamp, seq_num = analyze_queue.get()
            print(f"🔬 Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
            logging.info(f"Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
            
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"❌ File not found: {filepath}")
                continue
            
            sample_rate, data = wav.read(filepath)
            if len(data.shape) > 1:
                data = data.flatten()
                
            datum = Datum(audio_array=data, sample_rate=sample_rate)
            datum = pipeline.run(datum)
            
            # Use mock UART instead of real UART
            success = mock_uart.send_packet(process_timestamp, datum, seq_num)
            
            if success:
                rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
                status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0
                freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
                peak_freq_dec = datum.get_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK) or 0.0
                print(f"   ⚡ Analysis complete: RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz, DecFreq:{peak_freq_dec:.1f}Hz")
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
            
            print(f"📊 Status: Queue: {analyze_queue.qsize()}/{ANALYZE_QUEUE_SIZE}, "
                  f"Packets sent: {mock_uart.packet_count}")
                      
            logging.info(f"System status - Queue: {analyze_queue.qsize()}, "
                        f"Packets sent: {mock_uart.packet_count}")
                        
        except Exception as e:
            logging.error(f"Status monitor error: {e}")

def main():
    print("🧪 Audio Processing System - TEST MODE (IMPROVED)")
    print(f"📁 Processing files from: {AUDIO_DIR}")
    print(f"⏱️ Process interval: {PROCESS_INTERVAL} seconds")
    print(f"🔄 Loop processing: {LOOP_PROCESSING}")
    print(f"🎯 Target decimated rate: {DECIMATED_RATE} Hz")
    print(f"🚗 Events per cycle: {EVENTS_PER_CYCLE}")
    print(f"🔧 IMPROVEMENTS: Bandpass filtering (20-150 Hz), scipy.signal.resample, rolling buffer, find_peaks")
    
    logging.info("Audio Processing System - TEST MODE (IMPROVED) Starting")
    
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
    
    print("✅ Improved test system started successfully!")
    print("📊 Active threads:")
    print("   🎵 File processor (replacing recorder)")
    print("   🔬 Signal analyzer (IMPROVED with bandpass filtering & better peak detection)") 
    print("   📈 Status monitor")
    print("\n💻 Commands:")
    print("   - Press 's' + Enter for status")
    print("   - Press 'q' + Enter to quit")
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
                    print(f"\n📊 System Status:")
                    print(f"   Queue: {analyze_queue.qsize()}/{ANALYZE_QUEUE_SIZE}")
                    print(f"   Packets sent: {mock_uart.packet_count}")
                    print(f"   Files available: {len(get_audio_files())}")
                    print(f"   Loop mode: {LOOP_PROCESSING}")
                    print(f"   Process interval: {PROCESS_INTERVAL}s")
                    print(f"   Improvements: Bandpass (20-150 Hz), resample, rolling buffer, find_peaks\n")
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
        print("🔄 Stopping improved test system...")
        logging.info("Stopping improved test system")
        print("✅ Improved test system stopped cleanly")
        logging.info("Improved test system stopped")

if __name__ == "__main__":
    main()