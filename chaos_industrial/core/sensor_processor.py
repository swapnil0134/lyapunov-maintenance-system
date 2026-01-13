"""
Sensor Data Processing for Industrial Machine Failure Detection

This module handles multi-sensor data fusion, preprocessing, and validation
for industrial machinery monitoring systems.
"""

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.stats import zscore
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class SensorProcessor:
    """Multi-sensor data processing and fusion for industrial monitoring"""
    
    def __init__(self, sample_rate=1000, buffer_size=1000):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Sensor configuration
        self.sensors = {
            'accelerometer': {
                'range': [-50, 50],
                'sensitivity': 100,
                'units': 'g',
                'weight': 0.4
            },
            'vibration': {
                'range': [-20, 20],
                'sensitivity': 10,
                'units': 'mm/s',
                'weight': 0.3
            },
            'temperature': {
                'range': [0, 150],
                'sensitivity': 1,
                'units': '°C',
                'weight': 0.2
            },
            'acoustic': {
                'range': [0, 100],
                'sensitivity': 1,
                'units': 'dB',
                'weight': 0.1
            }
        }
        
        # Data buffers for each sensor
        self.data_buffers = {
            sensor: deque(maxlen=buffer_size) 
            for sensor in self.sensors.keys()
        }
        
        # Preprocessing parameters
        self.filter_config = {
            'accelerometer': {
                'highpass': 5,    # Remove DC component
                'lowpass': 500,   # Anti-aliasing
                'order': 4
            },
            'vibration': {
                'bandpass': [10, 1000],
                'order': 4
            },
            'temperature': {
                'lowpass': 1,     # Slow varying signal
                'order': 2
            },
            'acoustic': {
                'bandpass': [100, 5000],
                'order': 4
            }
        }
    
    def add_sensor_data(self, sensor_type, data):
        """
        Add new sensor data to buffer
        
        Args:
            sensor_type: Type of sensor (accelerometer, vibration, etc.)
            data: Sensor reading or array of readings
        """
        if sensor_type not in self.sensors:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        
        # Validate data range
        if np.isscalar(data):
            data = np.array([data])
        else:
            data = np.array(data)
        
        # Clip to valid range
        min_range, max_range = self.sensors[sensor_type]['range']
        data = np.clip(data, min_range, max_range)
        
        # Add to buffer
        for value in data:
            self.data_buffers[sensor_type].append(value)
    
    def preprocess_sensor_data(self, sensor_type, data):
        """
        Preprocess raw sensor data
        
        Args:
            sensor_type: Type of sensor
            data: Raw sensor data
            
        Returns:
            Preprocessed data
        """
        if sensor_type not in self.filter_config:
            return data
        
        config = self.filter_config[sensor_type]
        
        try:
            # Apply filtering based on sensor type
            if sensor_type == 'accelerometer':
                data = self._filter_accelerometer(data, config)
            elif sensor_type == 'vibration':
                data = self._filter_vibration(data, config)
            elif sensor_type == 'temperature':
                data = self._filter_temperature(data, config)
            elif sensor_type == 'acoustic':
                data = self._filter_acoustic(data, config)
            
            # Remove outliers using z-score
            if len(data) > 10:
                z_scores = np.abs(zscore(data))
                data = data[z_scores < 3]  # Remove outliers > 3 sigma
            
            # Apply smoothing if enough data
            if len(data) > 51:
                data = savgol_filter(data, window_length=51, polyorder=3)
            
            return data
            
        except:
            return data
    
    def _filter_accelerometer(self, data, config):
        """Apply accelerometer-specific filtering"""
        # High-pass to remove gravity/DC
        if config['highpass'] > 0:
            nyquist = self.sample_rate / 2
            high_freq = config['highpass'] / nyquist
            if high_freq < 1:
                b, a = butter(config['order'], high_freq, btype='high')
                data = filtfilt(b, a, data)
        
        # Low-pass for anti-aliasing
        if config['lowpass'] > 0:
            nyquist = self.sample_rate / 2
            low_freq = config['lowpass'] / nyquist
            if low_freq < 1:
                b, a = butter(config['order'], low_freq, btype='low')
                data = filtfilt(b, a, data)
        
        return data
    
    def _filter_vibration(self, data, config):
        """Apply vibration-specific band-pass filtering"""
        nyquist = self.sample_rate / 2
        low_freq = config['bandpass'][0] / nyquist
        high_freq = config['bandpass'][1] / nyquist
        
        if high_freq < 1:
            b, a = butter(config['order'], [low_freq, high_freq], btype='band')
            data = filtfilt(b, a, data)
        
        return data
    
    def _filter_temperature(self, data, config):
        """Apply temperature-specific low-pass filtering"""
        nyquist = self.sample_rate / 2
        low_freq = config['lowpass'] / nyquist
        
        if low_freq < 1:
            b, a = butter(config['order'], low_freq, btype='low')
            data = filtfilt(b, a, data)
        
        return data
    
    def _filter_acoustic(self, data, config):
        """Apply acoustic-specific band-pass filtering"""
        nyquist = self.sample_rate / 2
        low_freq = config['bandpass'][0] / nyquist
        high_freq = config['bandpass'][1] / nyquist
        
        if high_freq < 1:
            b, a = butter(config['order'], [low_freq, high_freq], btype='band')
            data = filtfilt(b, a, data)
        
        return data
    
    def get_sensor_data(self, sensor_type, window_size=None):
        """
        Get sensor data from buffer
        
        Args:
            sensor_type: Type of sensor
            window_size: Number of recent samples to return
            
        Returns:
            Sensor data array
        """
        if sensor_type not in self.data_buffers:
            return np.array([])
        
        buffer = self.data_buffers[sensor_type]
        if not buffer:
            return np.array([])
        
        data = np.array(buffer)
        
        if window_size is not None and window_size < len(data):
            data = data[-window_size:]
        
        return data
    
    def fuse_sensor_data(self, fusion_method='weighted_average'):
        """
        Fuse data from multiple sensors
        
        Args:
            fusion_method: Method for sensor fusion
            
        Returns:
            Fused sensor data
        """
        # Get available sensor data
        available_data = {}
        for sensor_type in self.sensors.keys():
            data = self.get_sensor_data(sensor_type)
            if len(data) > 0:
                # Normalize data based on sensor range
                min_range, max_range = self.sensors[sensor_type]['range']
                normalized_data = (data - min_range) / (max_range - min_range)
                available_data[sensor_type] = normalized_data
        
        if not available_data:
            return np.array([])
        
        # Find minimum length
        min_length = min(len(data) for data in available_data.values())
        if min_length == 0:
            return np.array([])
        
        # Reshape data to common length
        aligned_data = {}
        for sensor_type, data in available_data.items():
            if len(data) >= min_length:
                aligned_data[sensor_type] = data[-min_length:]
        
        # Apply fusion method
        if fusion_method == 'weighted_average':
            return self._weighted_average_fusion(aligned_data)
        elif fusion_method == 'principal_component':
            return self._principal_component_fusion(aligned_data)
        elif fusion_method == 'max_amplitude':
            return self._max_amplitude_fusion(aligned_data)
        else:
            return self._weighted_average_fusion(aligned_data)
    
    def _weighted_average_fusion(self, aligned_data):
        """Weighted average fusion based on sensor importance"""
        if not aligned_data:
            return np.array([])
        
        # Get common length
        min_length = min(len(data) for data in aligned_data.values())
        fused_data = np.zeros(min_length)
        total_weight = 0
        
        for sensor_type, data in aligned_data.items():
            weight = self.sensors[sensor_type]['weight']
            # Use last min_length samples
            sensor_contribution = data[-min_length:] * weight
            fused_data += sensor_contribution
            total_weight += weight
        
        if total_weight > 0:
            fused_data /= total_weight
        
        return fused_data
    
    def _principal_component_fusion(self, aligned_data):
        """Principal Component Analysis fusion"""
        if len(aligned_data) < 2:
            # Not enough sensors for PCA
            return self._weighted_average_fusion(aligned_data)
        
        try:
            # Create data matrix
            min_length = min(len(data) for data in aligned_data.values())
            data_matrix = np.zeros((min_length, len(aligned_data)))
            
            for i, (sensor_type, data) in enumerate(aligned_data.items()):
                data_matrix[:, i] = data[-min_length:]
            
            # Apply PCA
            cov_matrix = np.cov(data_matrix.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # Use principal component with largest eigenvalue
            principal_component = eigenvectors[:, np.argmax(eigenvalues)]
            fused_data = np.dot(data_matrix, principal_component)
            
            return fused_data
            
        except:
            return self._weighted_average_fusion(aligned_data)
    
    def _max_amplitude_fusion(self, aligned_data):
        """Maximum amplitude fusion (conservative approach)"""
        if not aligned_data:
            return np.array([])
        
        min_length = min(len(data) for data in aligned_data.values())
        fused_data = np.zeros(min_length)
        
        for i in range(min_length):
            max_values = []
            for data in aligned_data.values():
                if i < len(data):
                    max_values.append(abs(data[i]))
            
            if max_values:
                # Select sensor with maximum amplitude at each time point
                max_idx = np.argmax([abs(data[i]) for data in aligned_data.values()])
                sensor_names = list(aligned_data.keys())
                fused_data[i] = aligned_data[sensor_names[max_idx]][i]
        
        return fused_data
    
    def validate_sensor_health(self, sensor_type):
        """
        Validate sensor health and data quality
        
        Args:
            sensor_type: Type of sensor to validate
            
        Returns:
            Dictionary with health metrics
        """
        if sensor_type not in self.data_buffers:
            return {'healthy': False, 'error': 'Unknown sensor type'}
        
        buffer = self.data_buffers[sensor_type]
        if len(buffer) < 100:
            return {'healthy': False, 'error': 'Insufficient data'}
        
        data = np.array(buffer)
        
        # Check for data quality issues
        health_metrics = {
            'healthy': True,
            'error': None,
            'data_count': len(data),
            'mean_value': np.mean(data),
            'std_value': np.std(data),
            'min_value': np.min(data),
            'max_value': np.max(data)
        }
        
        # Check for stuck sensor (low variance)
        if np.std(data) < 0.01 * (np.max(data) - np.min(data)):
            health_metrics['healthy'] = False
            health_metrics['error'] = 'Sensor appears stuck (low variance)'
        
        # Check for out-of-range values
        min_range, max_range = self.sensors[sensor_type]['range']
        if np.any(data < min_range) or np.any(data > max_range):
            health_metrics['healthy'] = False
            health_metrics['error'] = 'Values out of sensor range'
        
        # Check for excessive noise (high frequency content)
        if len(data) > 20:
            # Simple noise check using high-frequency component
            diff_signal = np.diff(data)
            if len(diff_signal) > 0:
                noise_level = np.std(diff_signal)
                signal_level = np.std(data)
                
                if signal_level > 0 and noise_level / signal_level > 0.5:
                    health_metrics['warning'] = 'High noise level detected'
        
        return health_metrics
    
    def get_all_sensor_health(self):
        """Get health status for all sensors"""
        health_report = {}
        for sensor_type in self.sensors.keys():
            health_report[sensor_type] = self.validate_sensor_health(sensor_type)
        
        return health_report