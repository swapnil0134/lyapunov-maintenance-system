"""
Industrial Machine Failure Detection System

This module extends the chaos-based robot failure detection with industrial-specific
features including multi-mode failure detection, frequency analysis, and real-time
alerting for comprehensive machinery monitoring.
"""

import numpy as np
import pandas as pd
import serial
import time
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..core import FastChaosAnalyzer, FrequencyAnalyzer, SensorProcessor


class IndustrialFailureDetector:
    """Comprehensive industrial machine failure detection system"""
    
    def __init__(self, window_size=1000, sample_rate=1000, config=None):
        # Core parameters
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.update_interval = 50
        
        # Initialize core components
        self.chaos_analyzer = FastChaosAnalyzer()
        self.frequency_analyzer = FrequencyAnalyzer(sample_rate)
        self.sensor_processor = SensorProcessor(sample_rate, window_size)
        
        # Failure detection parameters
        self.global_threshold = 0.15
        self.alert_threshold = 0.7
        self.critical_threshold = 0.9
        
        # Data storage
        self.baseline_data = deque(maxlen=window_size)
        self.current_data = deque(maxlen=window_size)
        
        # Baseline metrics
        self.baseline_chaos = None
        self.baseline_frequency = None
        self.is_baseline_learned = False
        
        # Detection state
        self.status = "LEARNING BASELINE"
        self.anomaly_score = 0.0
        self.failure_modes = {}
        self.alert_cooldowns = {}
        
        # History tracking
        self.history = {
            'timestamp': [],
            'chaos_metrics': [],
            'frequency_features': [],
            'anomaly_score': [],
            'failure_modes': [],
            'status': []
        }
        
        # Configuration
        self.config = config or self._default_config()
    
    def _default_config(self):
        """Default configuration for industrial failure detection"""
        return {
            'failure_modes': {
                'bearing_wear': {
                    'enabled': True,
                    'weight': 0.3,
                    'threshold': 0.25,
                    'frequency_range': [500, 1000],
                    'chaos_indicators': ['lyapunov', 'correlation_dimension']
                },
                'unbalance': {
                    'enabled': True,
                    'weight': 0.25,
                    'threshold': 0.2,
                    'frequency_range': [0, 200],
                    'chaos_indicators': ['approximate_entropy']
                },
                'misalignment': {
                    'enabled': True,
                    'weight': 0.25,
                    'threshold': 0.2,
                    'frequency_range': [100, 500],
                    'chaos_indicators': ['lyapunov']
                },
                'looseness': {
                    'enabled': True,
                    'weight': 0.2,
                    'threshold': 0.3,
                    'frequency_range': [200, 800],
                    'chaos_indicators': ['correlation_dimension', 'approximate_entropy']
                }
            },
            'alerting': {
                'cooldown_period': 30,  # seconds
                'escalation_threshold': 0.8,
                'min_confidence': 0.6
            }
        }
    
    def add_sensor_data(self, sensor_type, data):
        """
        Add new sensor data to the system
        
        Args:
            sensor_type: Type of sensor (accelerometer, vibration, etc.)
            data: Sensor reading or array of readings
        """
        # Add to sensor processor
        self.sensor_processor.add_sensor_data(sensor_type, data)
        
        # Get fused data for main analysis
        fused_data = self.sensor_processor.fuse_sensor_data('weighted_average')
        
        if len(fused_data) > 0:
            self._process_new_data(fused_data[-1])  # Process latest value
    
    def add_accelerometer_sample(self, accel_x, accel_y, accel_z):
        """
        Add 3-axis accelerometer sample (legacy compatibility)
        
        Args:
            accel_x: X-axis acceleration
            accel_y: Y-axis acceleration  
            accel_z: Z-axis acceleration
        """
        # Calculate magnitude
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Add as accelerometer sensor data
        self.add_sensor_data('accelerometer', magnitude)
    
    def _process_new_data(self, new_sample):
        """Process new data sample through the detection pipeline"""
        # Store sample
        if not self.is_baseline_learned:
            self.baseline_data.append(new_sample)
        else:
            self.current_data.append(new_sample)
        
        # Process when we have enough data
        if not self.is_baseline_learned:
            if len(self.baseline_data) >= self.window_size:
                self._learn_baseline()
        else:
            if len(self.current_data) >= self.window_size:
                self._detect_failures()
    
    def _learn_baseline(self):
        """Learn baseline characteristics from healthy machine operation"""
        print("Learning baseline industrial failure metrics...")
        
        baseline_array = np.array(self.baseline_data)
        
        # Calculate baseline chaos metrics
        self.baseline_chaos = self.chaos_analyzer.calculate_all_metrics(
            baseline_array, emb_dim=3, delay=1
        )
        
        # Calculate baseline frequency features
        self.baseline_frequency = self.frequency_analyzer.extract_frequency_features(
            baseline_array
        )
        
        # Detect baseline failure patterns
        self.baseline_frequency['failure_patterns'] = self.frequency_analyzer.detect_failure_patterns(
            self.baseline_frequency
        )
        
        print(f"Baseline Chaos Metrics: {self.baseline_chaos}")
        print(f"Baseline Frequency Features: {self._summarize_frequency_features(self.baseline_frequency)}")
        
        self.is_baseline_learned = True
        self.status = "MONITORING"
        
        # Add to history
        self.history['timestamp'].append(datetime.now())
        self.history['chaos_metrics'].append(self.baseline_chaos)
        self.history['frequency_features'].append(self.baseline_frequency)
        self.history['anomaly_score'].append(0.0)
        self.history['failure_modes'].append({})
        self.history['status'].append(self.status)
    
    def _detect_failures(self):
        """Detect failures by comparing current metrics to baseline"""
        current_array = np.array(self.current_data)
        current_time = datetime.now()
        
        # Calculate current metrics
        current_chaos = self.chaos_analyzer.calculate_all_metrics(
            current_array, emb_dim=3, delay=1
        )
        current_frequency = self.frequency_analyzer.extract_frequency_features(
            current_array
        )
        
        # Detect failure patterns
        current_frequency['failure_patterns'] = self.frequency_analyzer.detect_failure_patterns(
            current_frequency
        )
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(
            current_chaos, current_frequency, self.baseline_chaos, self.baseline_frequency
        )
        
        # Classify failure modes
        failure_modes = self._classify_failure_modes(
            current_chaos, current_frequency, anomaly_score
        )
        
        # Determine status
        self.status = self._determine_status(anomaly_score, failure_modes)
        
        # Store results
        self.anomaly_score = anomaly_score
        self.failure_modes = failure_modes
        
        # Update history
        self.history['timestamp'].append(current_time)
        self.history['chaos_metrics'].append(current_chaos)
        self.history['frequency_features'].append(current_frequency)
        self.history['anomaly_score'].append(anomaly_score)
        self.history['failure_modes'].append(failure_modes)
        self.history['status'].append(self.status)
    
    def _calculate_anomaly_score(self, current_chaos, current_freq, baseline_chaos, baseline_freq):
        """Calculate composite anomaly score from chaos and frequency metrics"""
        score_components = []
        
        # Chaos metrics contributions
        for metric in ['lyapunov', 'correlation_dimension', 'approximate_entropy']:
            if metric in current_chaos and metric in baseline_chaos:
                current_val = current_chaos[metric]
                baseline_val = baseline_chaos[metric]
                
                if baseline_val > 0:
                    deviation = abs(current_val - baseline_val) / baseline_val
                    score_components.append(min(deviation, 2.0))  # Cap at 2.0
        
        # Frequency metrics contributions
        # High frequency content increase (bearing wear indicator)
        current_energy = current_freq.get('band_energy', {})
        baseline_energy = baseline_freq.get('band_energy', {})
        
        if 'high' in current_energy and 'high' in baseline_energy:
            if baseline_energy['high'] > 0:
                high_freq_increase = current_energy['high'] / baseline_energy['high']
                score_components.append(min(high_freq_increase - 1.0, 2.0))
        
        # Harmonic content changes
        current_harmonic = current_freq.get('harmonic_ratio', 0)
        baseline_harmonic = baseline_freq.get('harmonic_ratio', 0)
        
        harmonic_change = abs(current_harmonic - baseline_harmonic)
        score_components.append(harmonic_change)
        
        # Peak count changes (looseness indicator)
        current_peaks = current_freq.get('peak_count', 0)
        baseline_peaks = baseline_freq.get('peak_count', 0)
        
        if baseline_peaks > 0:
            peak_change = abs(current_peaks - baseline_peaks) / baseline_peaks
            score_components.append(min(peak_change, 2.0))
        
        # Calculate weighted average
        if score_components:
            return np.mean(score_components)
        else:
            return 0.0
    
    def _classify_failure_modes(self, current_chaos, current_freq, anomaly_score):
        """Classify specific failure modes based on metric patterns"""
        failure_modes = {}
        
        for mode_name, mode_config in self.config['failure_modes'].items():
            if not mode_config['enabled']:
                continue
            
            mode_score = 0.0
            confidence = 0.0
            
            # Frequency-based detection
            freq_patterns = current_freq.get('failure_patterns', {})
            if mode_name in freq_patterns:
                pattern = freq_patterns[mode_name]
                if pattern.get('detected', False):
                    mode_score += pattern.get('confidence', 0.0) * 0.6
                    confidence = max(confidence, pattern.get('confidence', 0.0))
            
            # Chaos-based detection
            for chaos_indicator in mode_config['chaos_indicators']:
                if chaos_indicator in current_chaos and chaos_indicator in self.baseline_chaos:
                    current_val = current_chaos[chaos_indicator]
                    baseline_val = self.baseline_chaos[chaos_indicator]
                    
                    if baseline_val > 0:
                        deviation = abs(current_val - baseline_val) / baseline_val
                        if deviation > 0.2:  # Significant change
                            mode_score += deviation * 0.3
                            confidence = max(confidence, min(deviation, 1.0))
            
            # Apply mode-specific weighting
            weighted_score = mode_score * mode_config['weight']
            
            failure_modes[mode_name] = {
                'detected': weighted_score > mode_config['threshold'],
                'score': weighted_score,
                'confidence': confidence,
                'severity': self._determine_severity(weighted_score, mode_config['threshold'])
            }
        
        return failure_modes
    
    def _determine_severity(self, score, threshold):
        """Determine failure severity level"""
        if score < threshold * 0.5:
            return 'normal'
        elif score < threshold:
            return 'minor'
        elif score < threshold * 1.5:
            return 'moderate'
        else:
            return 'severe'
    
    def _determine_status(self, anomaly_score, failure_modes):
        """Determine overall system status"""
        # Check for critical failures
        critical_failures = [
            mode for mode, info in failure_modes.items() 
            if info.get('severity') == 'severe'
        ]
        
        if critical_failures:
            return f"🚨 CRITICAL: {', '.join(critical_failures)}"
        
        # Check for any detected failures
        detected_failures = [
            mode for mode, info in failure_modes.items() 
            if info.get('detected', False)
        ]
        
        if detected_failures:
            moderate_failures = [
                mode for mode in detected_failures
                if failure_modes[mode].get('severity') == 'moderate'
            ]
            
            if moderate_failures:
                return f"⚠️ WARNING: {', '.join(moderate_failures)}"
            else:
                return f"⚡ MINOR: {', '.join(detected_failures)}"
        
        # Check anomaly level
        if anomaly_score > self.critical_threshold:
            return "🔥 HIGH ANOMALY"
        elif anomaly_score > self.alert_threshold:
            return "⚠️ MEDIUM ANOMALY"
        elif anomaly_score > self.global_threshold:
            return "⚡ LOW ANOMALY"
        else:
            return "✓ NORMAL"
    
    def _summarize_frequency_features(self, freq_features):
        """Summarize key frequency features for display"""
        summary = {
            'dominant_freq': freq_features.get('dominant_frequency', 0),
            'harmonic_ratio': freq_features.get('harmonic_ratio', 0),
            'high_freq_energy': freq_features.get('band_energy', {}).get('high', 0)
        }
        return summary
    
    def get_current_status(self):
        """Get current detection status"""
        return {
            'status': self.status,
            'anomaly_score': self.anomaly_score,
            'failure_modes': self.failure_modes,
            'baseline_learned': self.is_baseline_learned,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_results(self, filename='industrial_failure_detection.csv'):
        """Export detection results to CSV"""
        if not self.history['timestamp']:
            print("No data to export")
            return
        
        # Prepare data for export
        export_data = []
        for i in range(len(self.history['timestamp'])):
            row = {
                'timestamp': self.history['timestamp'][i].isoformat(),
                'anomaly_score': self.history['anomaly_score'][i],
                'status': self.history['status'][i]
            }
            
            # Add chaos metrics
            if i < len(self.history['chaos_metrics']):
                chaos = self.history['chaos_metrics'][i]
                for metric, value in chaos.items():
                    row[f'chaos_{metric}'] = value
            
            # Add frequency features
            if i < len(self.history['frequency_features']):
                freq = self.history['frequency_features'][i]
                for feature, value in freq.items():
                    if feature not in ['failure_patterns', 'peak_frequencies', 'peak_amplitudes']:
                        row[f'freq_{feature}'] = value
            
            # Add failure mode information
            if i < len(self.history['failure_modes']):
                failures = self.history['failure_modes'][i]
                for mode, info in failures.items():
                    row[f'failure_{mode}'] = info.get('detected', False)
                    row[f'failure_{mode}_score'] = info.get('score', 0)
                    row[f'failure_{mode}_severity'] = info.get('severity', 'normal')
            
            export_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"✓ Results exported to {filename}")
        return df