"""
Frequency Domain Analysis for Industrial Machine Failure Detection

This module provides comprehensive frequency analysis capabilities including
FFT, envelope analysis, harmonic detection, and band-specific energy calculations
optimized for industrial machinery monitoring.
"""

import numpy as np
from scipy.signal import welch, hilbert, butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')


class FrequencyAnalyzer:
    """Comprehensive frequency domain analysis for industrial machinery"""
    
    def __init__(self, sample_rate=1000):
        self.sample_rate = sample_rate
        self.freq_bands = {
            'very_low': (0, 50),      # Very low frequency (structural)
            'low': (50, 200),         # Low frequency (imbalance, misalignment)
            'mid': (200, 500),        # Mid frequency (gear mesh, bearings)
            'high': (500, 1000),      # High frequency (bearing defects)
            'very_high': (1000, 2000) # Very high frequency (early bearing damage)
        }
        
        # Industrial machinery characteristic frequencies
        self.failure_freq_ranges = {
            'unbalance': (0.8, 1.2),      # Around 1X rotation frequency
            'misalignment': (1.8, 2.2),   # Around 2X rotation frequency
            'bearing_outer': (2.5, 4.0),  # Bearing outer race frequencies
            'bearing_inner': (4.0, 6.5),  # Bearing inner race frequencies
            'bearing_cage': (0.2, 0.5),   # Bearing cage frequencies
            'gear_mesh': (8.0, 15.0)      # Gear mesh frequencies
        }
    
    def extract_frequency_features(self, data):
        """
        Extract comprehensive frequency domain features
        
        Args:
            data: Time series vibration data
            
        Returns:
            Dictionary of frequency features
        """
        if len(data) < 256:
            return self._empty_features()
        
        # Basic FFT analysis using numpy FFT instead of welch
        if len(data) < 128:
            return self._empty_features()
        
        # Apply window to reduce spectral leakage
        window = np.hanning(len(data))
        windowed_data = data * window
        
        # Compute FFT
        fft_data = fft(windowed_data)
        fft_freq = fftfreq(len(data), 1/self.sample_rate)
        
        # Take only positive frequencies
        positive_freq_idx = np.where(fft_freq > 0)[0]
        freqs = fft_freq[positive_freq_idx]
        psd = np.abs(fft_data[positive_freq_idx])**2
        
        # Peak detection
        if len(psd) > 0 and np.max(psd) > 0:
            peaks, properties = find_peaks(psd, height=np.max(psd) * 0.1, distance=5)
        else:
            peaks = np.array([])
            properties = {}
        
        # Calculate features
        features = {
            # Basic frequency characteristics
            'dominant_frequency': freqs[np.argmax(psd)] if len(psd) > 0 else 0.0,
            'dominant_amplitude': np.max(psd) if len(psd) > 0 else 0.0,
            'frequency_centroid': self._calculate_frequency_centroid(freqs, psd),
            'frequency_spread': self._calculate_frequency_spread(freqs, psd),
            
            # Harmonic analysis
            'harmonic_ratio': self._calculate_harmonic_ratio(freqs, psd, peaks),
            'total_harmonic_distortion': self._calculate_thd(freqs, psd),
            
            # Band energy distribution
            'band_energy': self._calculate_band_energy(freqs, psd),
            'energy_ratio': self._calculate_energy_ratios(freqs, psd),
            
            # Envelope analysis (for bearing defects)
            'envelope_features': self._envelope_analysis(data),
            
            # Statistical frequency features
            'spectral_kurtosis': self._calculate_spectral_kurtosis(psd),
            'spectral_skewness': self._calculate_spectral_skewness(psd),
            'spectral_rolloff': self._calculate_spectral_rolloff(freqs, psd),
            
            # Peak characteristics
            'peak_count': len(peaks),
            'peak_frequencies': freqs[peaks].tolist() if len(peaks) > 0 else [],
            'peak_amplitudes': psd[peaks].tolist() if len(peaks) > 0 else []
        }
        
        return features
    
    def _empty_features(self):
        """Return empty feature set for insufficient data"""
        return {
            'dominant_frequency': 0.0,
            'dominant_amplitude': 0.0,
            'frequency_centroid': 0.0,
            'frequency_spread': 0.0,
            'harmonic_ratio': 0.0,
            'total_harmonic_distortion': 0.0,
            'band_energy': {band: 0.0 for band in self.freq_bands.keys()},
            'energy_ratio': {},
            'envelope_features': {},
            'spectral_kurtosis': 0.0,
            'spectral_skewness': 0.0,
            'spectral_rolloff': 0.0,
            'peak_count': 0,
            'peak_frequencies': [],
            'peak_amplitudes': []
        }
    
    def _calculate_frequency_centroid(self, freqs, psd):
        """Calculate frequency centroid (center of mass)"""
        if np.sum(psd) == 0:
            return 0.0
        return np.sum(freqs * psd) / np.sum(psd)
    
    def _calculate_frequency_spread(self, freqs, psd):
        """Calculate frequency spread (standard deviation)"""
        centroid = self._calculate_frequency_centroid(freqs, psd)
        if np.sum(psd) == 0:
            return 0.0
        variance = np.sum(psd * (freqs - centroid) ** 2) / np.sum(psd)
        return np.sqrt(max(0, variance))
    
    def _calculate_harmonic_ratio(self, freqs, psd, peaks):
        """Calculate harmonic content ratio"""
        if len(peaks) < 2:
            return 0.0
        
        # Find fundamental frequency (lowest significant peak)
        fundamental_freq = freqs[peaks[0]]
        if fundamental_freq == 0:
            return 0.0
        
        # Look for harmonics (2x, 3x, 4x fundamental)
        harmonic_energy = 0.0
        total_energy = np.sum(psd)
        
        for harmonic in [2, 3, 4]:
            target_freq = fundamental_freq * harmonic
            # Find peak near harmonic frequency
            tolerance = 0.1 * target_freq
            harmonic_mask = np.abs(freqs - target_freq) < tolerance
            harmonic_energy += np.sum(psd[harmonic_mask])
        
        return harmonic_energy / total_energy if total_energy > 0 else 0.0
    
    def _calculate_thd(self, freqs, psd):
        """Calculate Total Harmonic Distortion"""
        if len(psd) == 0:
            return 0.0
        
        # Find fundamental (dominant frequency)
        fundamental_idx = np.argmax(psd)
        fundamental_power = psd[fundamental_idx]
        
        if fundamental_power == 0:
            return 0.0
        
        # Calculate harmonic power (excluding fundamental)
        harmonic_power = np.sum(psd) - fundamental_power
        
        return np.sqrt(harmonic_power) / np.sqrt(fundamental_power) if fundamental_power > 0 else 0.0
    
    def _calculate_band_energy(self, freqs, psd):
        """Calculate energy in different frequency bands"""
        band_energy = {}
        
        for band_name, (freq_min, freq_max) in self.freq_bands.items():
            band_mask = (freqs >= freq_min) & (freqs <= freq_max)
            band_energy[band_name] = np.sum(psd[band_mask])
        
        return band_energy
    
    def _calculate_energy_ratios(self, freqs, psd):
        """Calculate energy ratios between different bands"""
        band_energy = self._calculate_band_energy(freqs, psd)
        total_energy = sum(band_energy.values())
        
        if total_energy == 0:
            return {}
        
        ratios = {}
        # High-to-low frequency ratio (indicator of bearing problems)
        if band_energy['low'] > 0:
            ratios['high_to_low'] = band_energy['high'] / band_energy['low']
        
        # Mid-to-low frequency ratio (indicator of gear problems)
        if band_energy['low'] > 0:
            ratios['mid_to_low'] = band_energy['mid'] / band_energy['low']
        
        # Very high frequency content (early bearing damage)
        ratios['very_high_ratio'] = band_energy['very_high'] / total_energy
        
        return ratios
    
    def _envelope_analysis(self, data):
        """Perform envelope analysis for bearing defect detection"""
        if len(data) < 128:
            return {}
        
        try:
            # Apply band-pass filter for bearing frequencies (1-5 kHz)
            filtered = self._bandpass_filter(data, 1000, 5000)
            
            # Calculate envelope using Hilbert transform
            analytic_signal = hilbert(filtered)
            envelope = np.abs(analytic_signal)
            
            # FFT of envelope using numpy FFT
            if len(envelope) >= 64:
                # Apply window to envelope
                env_window = np.hanning(len(envelope))
                env_windowed = envelope * env_window
                
                # Compute FFT
                env_fft = fft(env_windowed)
                env_freqs = fftfreq(len(envelope), 1/self.sample_rate)
                
                # Take only positive frequencies
                positive_env_idx = np.where(env_freqs > 0)[0]
                env_freqs = env_freqs[positive_env_idx]
                env_psd = np.abs(env_fft[positive_env_idx])**2
                
                # Find envelope peaks
                if len(env_psd) > 0 and np.max(env_psd) > 0:
                    env_peaks, _ = find_peaks(env_psd, height=np.max(env_psd) * 0.1)
                else:
                    env_peaks = np.array([])
            else:
                env_freqs = np.array([])
                env_psd = np.array([])
                env_peaks = np.array([])
            
            envelope_features = {
                'envelope_rms': np.sqrt(np.mean(envelope ** 2)),
                'envelope_kurtosis': kurtosis(envelope),
                'dominant_envelope_frequency': env_freqs[np.argmax(env_psd)] if len(env_psd) > 0 else 0.0,
                'envelope_peak_count': len(env_peaks),
                'envelope_peak_frequencies': env_freqs[env_peaks].tolist() if len(env_peaks) > 0 else []
            }
            
            return envelope_features
            
        except:
            return {}
    
    def _bandpass_filter(self, data, low_freq, high_freq, order=4):
        """Apply band-pass filter"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if high >= 1.0:
            # High-pass only
            b, a = butter(order, low, btype='high')
        else:
            # Band-pass
            b, a = butter(order, [low, high], btype='band')
        
        return filtfilt(b, a, data)
    
    def _calculate_spectral_kurtosis(self, psd):
        """Calculate spectral kurtosis"""
        if len(psd) < 2:
            return 0.0
        return kurtosis(psd)
    
    def _calculate_spectral_skewness(self, psd):
        """Calculate spectral skewness"""
        if len(psd) < 2:
            return 0.0
        return skew(psd)
    
    def _calculate_spectral_rolloff(self, freqs, psd, rolloff_percent=0.85):
        """Calculate spectral rolloff frequency"""
        if len(psd) == 0:
            return 0.0
        
        cumulative_energy = np.cumsum(psd)
        total_energy = cumulative_energy[-1]
        
        if total_energy == 0:
            return 0.0
        
        rolloff_threshold = rolloff_percent * total_energy
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
        
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1]
    
    def detect_failure_patterns(self, freq_features, rotation_freq=None):
        """
        Detect specific failure patterns based on frequency characteristics
        
        Args:
            freq_features: Frequency features dictionary
            rotation_freq: Known rotation frequency (Hz)
            
        Returns:
            Dictionary of failure pattern indicators
        """
        patterns = {}
        
        # Unbalance detection (dominant 1X frequency)
        if rotation_freq:
            freq_tolerance = 0.1 * rotation_freq
            fundamental_mask = np.abs(np.array(freq_features['peak_frequencies']) - rotation_freq) < freq_tolerance
            if np.any(fundamental_mask):
                patterns['unbalance'] = {
                    'detected': True,
                    'confidence': min(1.0, freq_features['harmonic_ratio'] * 2),
                    'dominant_1x': True
                }
            else:
                patterns['unbalance'] = {'detected': False, 'confidence': 0.0}
        
        # Misalignment detection (2X harmonic)
        if rotation_freq and len(freq_features['peak_frequencies']) > 0:
            freq_2x = 2 * rotation_freq
            freq_tolerance = 0.1 * freq_2x
            harmonic_2x_mask = np.abs(np.array(freq_features['peak_frequencies']) - freq_2x) < freq_tolerance
            if np.any(harmonic_2x_mask):
                patterns['misalignment'] = {
                    'detected': True,
                    'confidence': min(1.0, freq_features['harmonic_ratio'] * 1.5),
                    'dominant_2x': True
                }
            else:
                patterns['misalignment'] = {'detected': False, 'confidence': 0.0}
        
        # Bearing wear detection (high frequency content)
        energy_ratios = freq_features.get('energy_ratio', {})
        high_to_low = energy_ratios.get('high_to_low', 0.0)
        very_high_ratio = energy_ratios.get('very_high_ratio', 0.0)
        
        bearing_score = (high_to_low * 0.6 + very_high_ratio * 0.4)
        patterns['bearing_wear'] = {
            'detected': bearing_score > 0.3,
            'confidence': min(1.0, bearing_score * 2),
            'high_freq_content': high_to_low,
            'very_high_freq_content': very_high_ratio
        }
        
        # Looseness detection (multiple harmonics)
        peak_count = freq_features['peak_count']
        harmonic_ratio = freq_features['harmonic_ratio']
        
        looseness_score = (min(1.0, peak_count / 10) * 0.5 + harmonic_ratio * 0.5)
        patterns['looseness'] = {
            'detected': looseness_score > 0.4,
            'confidence': looseness_score,
            'peak_count': peak_count,
            'harmonic_content': harmonic_ratio
        }
        
        return patterns