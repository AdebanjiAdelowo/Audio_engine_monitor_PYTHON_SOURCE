#!/usr/bin/env python3
"""
Windows File Processing Script for Audio Analysis
Processes existing .wav files in AUDIO_DIR without real-time recording or UART
"""

from abc import ABC, abstractmethod
from enum import Enum
import os
import time
import glob
import struct
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
from scipy.signal import decimate, find_peaks, filtfilt, butter, resample, stft, istft
import logging
import matplotlib.pyplot as plt
import json
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

# Windows-specific settings
AUDIO_DIR = "./recordings"  # Change this to your actual recordings directory
LOG_FILE = "audio_analysis.log"
DECIMATED_RATE = 500
EVENTS_PER_CYCLE = 2
FFT_BUFFER_SECONDS = 10

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also print to console
    ]
)

# Create recordings directory if it doesn't exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# Enums (same as original)
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

# Datum Class (same as original)
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

    def get_raw_datum(self, key: RawDatumKey) -> Any:
        return self.__raw_datum.get(key)

    def get_raw_datum_keys(self) -> List[RawDatumKey]:
        return list(self.__raw_datum.keys())

    def add_derived_data(self, key: DerivedDataKey, value: Any) -> None:
        self.__derived_data[key] = value

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

# Filter Classes (same as corrected original)
class Filter(ABC):
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

class Resample(Filter):
    def __init__(self, new_rate: int) -> None:
        self.__new_rate = new_rate

    def get_new_rate(self) -> int:
        return self.__new_rate

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        resampled_length = int(len(signal) * self.__new_rate / sample_rate)
        return resample(signal, resampled_length)

# FFT Class (simplified)
class FFT:
    def __init__(self, fs=44100):
        self.__fs = fs
        self.__freq = None

    def get_fs(self):
        return self.__fs

    def set_fs(self, fs: int):
        self.__fs = fs

    def get_freq(self):
        return self.__freq

    def make_spectrum(self, time_series: np.ndarray):
        yf = fft(time_series)
        self.__freq = fftfreq(len(time_series), 1 / self.__fs)
        return yf

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

    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        signal = signal - np.mean(signal)
        filtered_signal = self.__bp_filter.apply_filter(signal, sample_rate)
        if self.__fft is None:
            self.__fft = FFT(sample_rate)
        elif self.__fft.get_fs() != sample_rate:
            self.__fft.set_fs(sample_rate)
        freq_signal = self.__fft.make_spectrum(filtered_signal)
        power_spectrum = np.abs(freq_signal[:len(freq_signal) // 2])
        peak_indices = find_peaks(power_spectrum, distance=sample_rate//100)[0]
        if len(peak_indices) == 0:
            return filtered_signal
        peak_index = peak_indices[np.argmax(power_spectrum[peak_indices])]
        peak_frequency = abs(self.__fft.get_freq()[peak_index])
        self.set_peak_freq(peak_frequency)
        return filtered_signal

# Feature Extraction Classes
class FeatureExtraction(ABC):
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

# File Processing Functions
def process_single_file(filepath, pipeline, verbose=True):
    """Process a single audio file"""
    try:
        if verbose:
            print(f"Processing: {os.path.basename(filepath)}")
        
        # Load audio file
        sample_rate, data = wav.read(filepath)
        
        # Handle stereo files
        if len(data.shape) > 1:
            data = data.flatten()
        
        # Create datum object
        datum = Datum(DatumType.AUDIO, audio_array=data, sample_rate=sample_rate)
        
        # Run through analysis pipeline
        datum = pipeline.run(datum)
        
        # Extract results
        results = {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'timestamp': os.path.getmtime(filepath),
            'sample_rate': sample_rate,
            'duration_seconds': len(data) / sample_rate,
            'rpm': datum.get_derived_data(DerivedDataKey.RPM) or 0.0,
            'engine_state': datum.get_derived_data(DerivedDataKey.ENGINE_STATE) or 0,
            'frequency_peak': datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0,
            'decimated_rate': datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE) or sample_rate
        }
        
        if verbose:
            print(f"  ✓ RPM: {results['rpm']:.1f}")
            print(f"  ✓ Engine State: {'ON' if results['engine_state'] else 'OFF'}")
            print(f"  ✓ Peak Frequency: {results['frequency_peak']:.1f} Hz")
            print(f"  ✓ Duration: {results['duration_seconds']:.1f}s")
            print("-" * 60)
        
        return results
        
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        logging.error(f"Error processing {filepath}: {e}")
        return None

def batch_process_files(audio_dir=None, pattern="*.wav", save_results=True, create_plots=True):
    """Process all audio files in directory"""
    
    if audio_dir is None:
        audio_dir = AUDIO_DIR
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline()
    pipeline.add_block(Decimation(target_rate=DECIMATED_RATE))
    pipeline.add_block(FrequencyPeakFinder(buffer_seconds=FFT_BUFFER_SECONDS))
    pipeline.add_output_block(RPM(events_per_crankshaft_cycle=EVENTS_PER_CYCLE))
    
    # Find all audio files
    file_pattern = os.path.join(audio_dir, pattern)
    audio_files = glob.glob(file_pattern)
    audio_files.sort()
    
    if not audio_files:
        print(f"❌ No audio files found in {audio_dir} matching pattern {pattern}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Contents of {audio_dir}:")
        try:
            for item in os.listdir(audio_dir):
                print(f"  - {item}")
        except FileNotFoundError:
            print(f"  Directory {audio_dir} does not exist!")
        return []
    
    print(f"🎵 Found {len(audio_files)} audio files to process")
    print("=" * 70)
    
    all_results = []
    start_time = time.time()
    
    for i, filepath in enumerate(audio_files, 1):
        print(f"[{i:3d}/{len(audio_files)}] ", end="")
        result = process_single_file(filepath, pipeline)
        
        if result:
            all_results.append(result)
    
    end_time = time.time()
    print(f"\n✅ Processed {len(all_results)} files in {end_time - start_time:.1f} seconds")
    
    if save_results and all_results:
        # Save to CSV
        df = pd.DataFrame(all_results)
        csv_filename = f"windows_audio_analysis_{int(time.time())}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"💾 Results saved to: {csv_filename}")
        
        # Save to JSON
        json_filename = f"windows_audio_analysis_{int(time.time())}.json"
        with open(json_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Results saved to: {json_filename}")
        
        # Print summary statistics
        print("\n" + "=" * 70)
        print("📊 SUMMARY STATISTICS:")
        print(f"Total files processed: {len(all_results)}")
        print(f"Average RPM: {df['rpm'].mean():.1f}")
        print(f"Max RPM: {df['rpm'].max():.1f}")
        print(f"Min RPM: {df['rpm'].min():.1f}")
        print(f"Engine ON recordings: {df['engine_state'].sum()} / {len(df)}")
        print(f"Average peak frequency: {df['frequency_peak'].mean():.1f} Hz")
        
        # Create plots if requested
        if create_plots:
            create_analysis_plots(df)
    
    return all_results

def create_analysis_plots(df):
    """Create visualization plots"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # RPM over time
        ax1.plot(df.index, df['rpm'], 'b-', linewidth=2)
        ax1.set_title('RPM Over Time (File Order)')
        ax1.set_xlabel('File Index')
        ax1.set_ylabel('RPM')
        ax1.grid(True, alpha=0.3)
        
        # RPM histogram
        ax2.hist(df['rpm'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('RPM Distribution')
        ax2.set_xlabel('RPM')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Frequency peak over time
        ax3.plot(df.index, df['frequency_peak'], 'r-', linewidth=2)
        ax3.set_title('Frequency Peak Over Time')
        ax3.set_xlabel('File Index')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.grid(True, alpha=0.3)
        
        # Engine state
        engine_counts = df['engine_state'].value_counts()
        ax4.pie(engine_counts.values, labels=['OFF', 'ON'], autopct='%1.1f%%', 
                colors=['red', 'green'], startangle=90)
        ax4.set_title('Engine State Distribution')
        
        plt.tight_layout()
        plot_filename = f"windows_audio_analysis_plots_{int(time.time())}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📈 Plots saved to: {plot_filename}")
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

def main():
    """Main function for Windows file processing"""
    print("🎵 Windows Audio File Processor")
    print("=" * 50)
    
    # Check if recordings directory exists
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Directory {AUDIO_DIR} does not exist!")
        print("Please create the directory and place your .wav files there, or update AUDIO_DIR variable.")
        return
    
    # List available files
    wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    if not wav_files:
        print(f"❌ No .wav files found in {AUDIO_DIR}")
        print("Please place your audio recordings in the directory.")
        return
    
    print(f"📁 Using recordings directory: {os.path.abspath(AUDIO_DIR)}")
    print(f"🎵 Found {len(wav_files)} .wav files")
    
    # Process all files
    results = batch_process_files()
    
    if results:
        print("\n✅ Processing completed successfully!")
        print("Check the generated CSV, JSON, and plot files for detailed results.")
    else:
        print("\n❌ No files were processed successfully.")

if __name__ == "__main__":
    main()