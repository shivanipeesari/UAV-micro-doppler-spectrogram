"""
Spectrogram Module
==================
This module handles generation and visualization of spectrograms
from raw radar signals using Short-Time Fourier Transform (STFT).

Author: B.Tech Major Project
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window
from scipy import signal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectrogramGenerator:
    """
    Generates micro-Doppler spectrograms from radar signals using STFT.
    
    This class handles:
    - Short-Time Fourier Transform (STFT) computation
    - Spectrogram visualization and saving
    - Signal preprocessing before STFT
    - Doppler frequency extraction
    """
    
    def __init__(self, sampling_rate=1000, n_fft=512):
        """
        Initialize SpectrogramGenerator.
        
        Args:
            sampling_rate (int): Sampling rate of the radar signal (Hz)
            n_fft (int): FFT window length
        """
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        logger.info(f"Initialized SpectrogramGenerator with sr={sampling_rate}, n_fft={n_fft}")
    
    def compute_stft(self, signal_data, window='hann', nperseg=256, noverlap=None):
        """
        Compute Short-Time Fourier Transform of the signal.
        
        Args:
            signal_data (np.ndarray): Input radar signal
            window (str): Window function ('hann', 'hamming', 'blackman')
            nperseg (int): Length of each segment for STFT
            noverlap (int): Number of overlapping samples between segments
        
        Returns:
            tuple: (f, t, Sxx) - frequencies, times, spectrogram magnitude
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Compute STFT
        f, t, Sxx = stft(
            signal_data,
            fs=self.sampling_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=self.n_fft
        )
        
        # Convert to magnitude (dB scale)
        Sxx_magnitude = np.abs(Sxx)
        Sxx_db = 20 * np.log10(Sxx_magnitude + 1e-10)  # Add small value to avoid log(0)
        
        logger.info(f"STFT computed: shape={Sxx_db.shape}")
        logger.info(f"Frequency range: {f[0]:.2f} to {f[-1]:.2f} Hz")
        logger.info(f"Time range: {t[0]:.4f} to {t[-1]:.4f} seconds")
        
        return f, t, Sxx_db
    
    def generate_spectrogram_image(self, signal_data, output_path=None,
                                   cmap='viridis', figsize=(10, 6)):
        """
        Generate and visualize spectrogram from radar signal.
        
        Args:
            signal_data (np.ndarray): Input radar signal
            output_path (str): Path to save the spectrogram image
            cmap (str): Colormap for visualization
            figsize (tuple): Figure size
        
        Returns:
            tuple: (figure, axes, spectrogram_data)
        """
        f, t, Sxx = self.compute_stft(signal_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot spectrogram
        pcolormesh = ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap=cmap)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Micro-Doppler Spectrogram')
        
        # Add colorbar
        cbar = fig.colorbar(pcolormesh, ax=ax)
        cbar.set_label('Magnitude (dB)')
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            logger.info(f"Spectrogram saved to {output_path}")
        
        return fig, ax, Sxx
    
    def preprocess_signal(self, signal_data, remove_dc=True, normalize=True):
        """
        Preprocess radar signal before STFT.
        
        Args:
            signal_data (np.ndarray): Input signal
            remove_dc (bool): Remove DC component
            normalize (bool): Normalize signal
        
        Returns:
            np.ndarray: Preprocessed signal
        """
        processed = signal_data.copy()
        
        # Remove DC component
        if remove_dc:
            processed = processed - np.mean(processed)
            logger.info("Removed DC component from signal")
        
        # Normalize
        if normalize:
            std = np.std(processed)
            if std > 0:
                processed = processed / std
            logger.info("Normalized signal")
        
        return processed
    
    def apply_window(self, signal_data, window_type='hann'):
        """
        Apply window function to signal to reduce spectral leakage.
        
        Args:
            signal_data (np.ndarray): Input signal
            window_type (str): Type of window ('hann', 'hamming', 'blackman')
        
        Returns:
            np.ndarray: Windowed signal
        """
        window = get_window(window_type, len(signal_data))
        windowed = signal_data * window
        return windowed
    
    def extract_micro_doppler_signature(self, spectrogram, doppler_range=None):
        """
        Extract micro-Doppler signature from spectrogram.
        
        Micro-Doppler is the small Doppler shifts caused by the motion of
        micro-components (propellers, wings) of the target.
        
        Args:
            spectrogram (np.ndarray): STFT spectrogram
            doppler_range (tuple): (min_freq, max_freq) to extract
        
        Returns:
            np.ndarray: Extracted micro-Doppler signature
        """
        # Simple extraction: take frequency components above a threshold
        threshold = np.percentile(spectrogram, 50)
        signature = spectrogram.copy()
        signature[signature < threshold] = 0
        
        logger.info(f"Extracted micro-Doppler signature: shape={signature.shape}")
        return signature
    
    def generate_synthetic_uav_signal(self, duration=1.0, base_frequency=100):
        """
        Generate synthetic UAV radar signal with micro-Doppler characteristics.
        
        UAVs have distinct micro-Doppler signatures from propeller rotation.
        This creates a signal with multiple frequency components.
        
        Args:
            duration (float): Signal duration in seconds
            base_frequency (float): Base Doppler frequency (Hz)
        
        Returns:
            np.ndarray: Synthetic UAV signal
        """
        t = np.arange(0, duration, 1/self.sampling_rate)
        
        # UAV signal: combination of base frequency and harmonics from propellers
        # Propeller frequencies are typically 100-300 Hz
        signal_uav = (
            0.8 * np.sin(2 * np.pi * base_frequency * t) +  # Base
            0.4 * np.sin(2 * np.pi * (base_frequency * 2) * t) +  # 2nd harmonic
            0.2 * np.sin(2 * np.pi * (base_frequency * 3) * t) +  # 3rd harmonic
            0.1 * np.random.randn(len(t))  # Added noise
        )
        
        logger.info(f"Generated synthetic UAV signal: duration={duration}s, length={len(signal_uav)}")
        return signal_uav
    
    def generate_synthetic_bird_signal(self, duration=1.0, base_frequency=50):
        """
        Generate synthetic bird radar signal with micro-Doppler characteristics.
        
        Birds have different micro-Doppler signatures with lower frequencies
        from wing flapping.
        
        Args:
            duration (float): Signal duration in seconds
            base_frequency (float): Base Doppler frequency (Hz)
        
        Returns:
            np.ndarray: Synthetic bird signal
        """
        t = np.arange(0, duration, 1/self.sampling_rate)
        
        # Bird signal: wing flapping frequency (5-20 Hz) modulated on base Doppler
        wing_flap_freq = 10  # Hz
        signal_bird = (
            0.7 * np.sin(2 * np.pi * base_frequency * t) +  # Base (Doppler shift)
            0.5 * np.sin(2 * np.pi * wing_flap_freq * t) * np.sin(2 * np.pi * base_frequency * t) +  # Wing modulation
            0.1 * np.random.randn(len(t))  # Added noise
        )
        
        logger.info(f"Generated synthetic bird signal: duration={duration}s, length={len(signal_bird)}")
        return signal_bird


def main():
    """
    Example usage of SpectrogramGenerator.
    """
    # Initialize generator
    generator = SpectrogramGenerator(sampling_rate=1000, n_fft=512)
    
    # Generate synthetic signals
    print("Generating synthetic signals...")
    uav_signal = generator.generate_synthetic_uav_signal(duration=1.0, base_frequency=150)
    bird_signal = generator.generate_synthetic_bird_signal(duration=1.0, base_frequency=80)
    
    # Preprocess signals
    print("Preprocessing signals...")
    uav_processed = generator.preprocess_signal(uav_signal)
    bird_processed = generator.preprocess_signal(bird_signal)
    
    # Generate spectrograms
    print("Generating spectrograms...")
    fig_uav, ax_uav, spec_uav = generator.generate_spectrogram_image(
        uav_processed,
        output_path="uav_spectrogram.png",
        cmap='magma'
    )
    
    fig_bird, ax_bird, spec_bird = generator.generate_spectrogram_image(
        bird_processed,
        output_path="bird_spectrogram.png",
        cmap='magma'
    )
    
    print(f"UAV spectrogram shape: {spec_uav.shape}")
    print(f"Bird spectrogram shape: {spec_bird.shape}")
    
    print("Spectrograms generated successfully!")
    print("Plots saved as uav_spectrogram.png and bird_spectrogram.png")


if __name__ == "__main__":
    main()
