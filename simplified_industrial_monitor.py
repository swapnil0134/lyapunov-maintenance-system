#!/usr/bin/env python3
"""
Simplified Industrial Machine Failure Detection System

A working version focused on core chaos-based failure detection
without complex signal processing issues.
"""

import numpy as np
import pandas as pd
import serial
import time
import argparse
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class SimplifiedIndustrialDetector:
    """Simplified industrial failure detection using chaos metrics"""
    
    def __init__(self, window_size=500, sample_rate=1000):
        self.window_size = window_size
        self.sample_rate = sample_rate
        
        # Data storage
        self.baseline_data = deque(maxlen=window_size)
        self.current_data = deque(maxlen=window_size)
        
        # Baseline metrics
        self.baseline_chaos = None
        self.is_baseline_learned = False
        
        # Detection state
        self.status = "LEARNING BASELINE"
        self.anomaly_score = 0.0
        
        # History
        self.history = {
            'timestamp': [],
            'lyapunov': [],
            'correlation_dim': [],
            'entropy': [],
            'anomaly_score': [],
            'status': []
        }
        
        # Thresholds
        self.lyapunov_threshold = 0.15
        self.correlation_dim_threshold = 0.3
        self.entropy_threshold = 0.2
        self.global_threshold = 0.25
    
    def add_accelerometer_sample(self, accel_x, accel_y, accel_z):
        """Add 3-axis accelerometer sample"""
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        if not self.is_baseline_learned:
            self.baseline_data.append(magnitude)
            if len(self.baseline_data) >= self.window_size:
                self._learn_baseline()
        else:
            self.current_data.append(magnitude)
            if len(self.current_data) >= self.window_size:
                self._detect_anomaly()
    
    def _learn_baseline(self):
        """Learn baseline chaos metrics"""
        data = np.array(self.baseline_data)
        
        print("Learning baseline chaos metrics...")
        self.baseline_chaos = self._calculate_chaos_metrics(data)
        
        print(f"Baseline Lyapunov: {self.baseline_chaos['lyapunov']:.4f}")
        print(f"Baseline Correlation Dim: {self.baseline_chaos['correlation_dim']:.4f}")
        print(f"Baseline Entropy: {self.baseline_chaos['entropy']:.4f}")
        
        self.is_baseline_learned = True
        self.status = "MONITORING"
    
    def _detect_anomaly(self):
        """Detect anomalies using chaos metrics"""
        data = np.array(self.current_data)
        current_chaos = self._calculate_chaos_metrics(data)
        
        # Calculate deviations
        lyapunov_dev = abs(current_chaos['lyapunov'] - self.baseline_chaos['lyapunov'])
        corr_dev = abs(current_chaos['correlation_dim'] - self.baseline_chaos['correlation_dim'])
        entropy_dev = abs(current_chaos['entropy'] - self.baseline_chaos['entropy'])
        
        # Normalize deviations
        lyapunov_norm = min(lyapunov_dev / self.lyapunov_threshold, 2.0)
        corr_norm = min(corr_dev / self.correlation_dim_threshold, 2.0)
        entropy_norm = min(entropy_dev / self.entropy_threshold, 2.0)
        
        # Calculate composite anomaly score
        self.anomaly_score = (lyapunov_norm + corr_norm + entropy_norm) / 3.0
        
        # Determine status
        if self.anomaly_score > self.global_threshold * 2:
            self.status = "🚨 CRITICAL FAILURE"
        elif self.anomaly_score > self.global_threshold * 1.5:
            self.status = "⚠️ SEVERE ANOMALY"
        elif self.anomaly_score > self.global_threshold:
            self.status = "⚡ ANOMALY DETECTED"
        else:
            self.status = "✓ NORMAL"
        
        # Store in history
        self.history['timestamp'].append(datetime.now())
        self.history['lyapunov'].append(current_chaos['lyapunov'])
        self.history['correlation_dim'].append(current_chaos['correlation_dim'])
        self.history['entropy'].append(current_chaos['entropy'])
        self.history['anomaly_score'].append(self.anomaly_score)
        self.history['status'].append(self.status)
    
    def _calculate_chaos_metrics(self, data):
        """Calculate simplified chaos metrics"""
        if len(data) < 100:
            return {'lyapunov': 0.0, 'correlation_dim': 2.0, 'entropy': 0.0}
        
        # Simplified Lyapunov exponent estimation
        lyapunov = self._estimate_lyapunov_simplified(data)
        
        # Simplified correlation dimension
        corr_dim = self._estimate_correlation_dimension_simplified(data)
        
        # Simplified entropy
        entropy = self._estimate_entropy_simplified(data)
        
        return {
            'lyapunov': lyapunov,
            'correlation_dim': corr_dim,
            'entropy': entropy
        }
    
    def _estimate_lyapunov_simplified(self, data, emb_dim=3, delay=1):
        """Simplified Lyapunov exponent estimation"""
        n = len(data)
        if n < emb_dim * delay + 10:
            return 0.0
        
        # Create embedding
        N = n - (emb_dim - 1) * delay
        embedded = np.zeros((N, emb_dim))
        for i in range(emb_dim):
            embedded[:, i] = data[i * delay : i * delay + N]
        
        # Find nearest neighbors
        divergence_rates = []
        for i in range(min(50, N - 10)):
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            distances[max(0, i-5):i+6] = np.inf  # Exclude temporal neighbors
            
            if np.min(distances) < np.inf:
                nearest_idx = np.argmin(distances)
                
                # Track divergence for next few steps
                for k in range(1, min(10, N - max(i, nearest_idx))):
                    d_ik = np.linalg.norm(embedded[i + k] - embedded[nearest_idx + k])
                    if d_ik > 0:
                        divergence_rates.append(np.log(d_ik))
                        break
        
        if len(divergence_rates) > 5:
            return max(0.0, np.mean(divergence_rates))
        return 0.0
    
    def _estimate_correlation_dimension_simplified(self, data, emb_dim=5):
        """Simplified correlation dimension estimation"""
        n = len(data)
        N = n - (emb_dim - 1)
        
        if N < 50:
            return 2.0
        
        # Create embedding
        embedded = np.zeros((N, emb_dim))
        for i in range(emb_dim):
            embedded[:, i] = data[i : i + N]
        
        # Sample points for efficiency
        sample_size = min(N, 100)
        sample_indices = np.random.choice(N, sample_size, replace=False)
        sample_embedded = embedded[sample_indices]
        
        # Calculate distances
        distances = []
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                distances.append(np.linalg.norm(sample_embedded[i] - sample_embedded[j]))
        
        if len(distances) == 0:
            return 2.0
        
        distances = np.array(distances)
        
        # Estimate dimension from distance distribution
        percentiles = [25, 50, 75]
        p_values = []
        for p in percentiles:
            r = np.percentile(distances, p)
            if r > 0:
                c = np.sum(distances < r) / len(distances)
                if c > 0:
                    p_values.append(np.log(c) / np.log(r))
        
        if len(p_values) > 1:
            return max(0.5, min(8.0, -np.mean(p_values)))
        return 2.0
    
    def _estimate_entropy_simplified(self, data, m=2):
        """Simplified approximate entropy"""
        N = len(data)
        if N < m + 1:
            return 0.0
        
        # Create patterns
        patterns_m = []
        patterns_m1 = []
        
        for i in range(N - m):
            pattern = tuple(data[i:i+m])
            patterns_m.append(pattern)
        
        for i in range(N - m - 1):
            pattern = tuple(data[i:i+m+1])
            patterns_m1.append(pattern)
        
        # Calculate pattern frequencies
        from collections import Counter
        
        count_m = Counter(patterns_m)
        count_m1 = Counter(patterns_m1)
        
        N_m = len(patterns_m)
        N_m1 = len(patterns_m1)
        
        # Calculate entropy
        phi_m = 0
        for pattern, count in count_m.items():
            prob = count / N_m
            if prob > 0:
                phi_m += prob * np.log(prob)
        
        phi_m1 = 0
        for pattern, count in count_m1.items():
            prob = count / N_m1
            if prob > 0:
                phi_m1 += prob * np.log(prob)
        
        return max(0.0, phi_m - phi_m1)
    
    def get_current_status(self):
        """Get current detection status"""
        return {
            'status': self.status,
            'anomaly_score': self.anomaly_score,
            'baseline_learned': self.is_baseline_learned,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_results(self, filename='simplified_failure_detection.csv'):
        """Export results to CSV"""
        if not self.history['timestamp']:
            print("No data to export")
            return
        
        df = pd.DataFrame({
            'timestamp': self.history['timestamp'],
            'lyapunov': self.history['lyapunov'],
            'correlation_dimension': self.history['correlation_dim'],
            'entropy': self.history['entropy'],
            'anomaly_score': self.history['anomaly_score'],
            'status': self.history['status']
        })
        
        df.to_csv(filename, index=False)
        print(f"✓ Results exported to {filename}")
        return df


class SimplifiedIndustrialMonitor:
    """Main monitoring application"""
    
    def __init__(self, simulate=False, serial_port='/dev/ttyUSB0', baud_rate=115200):
        self.simulate = simulate
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        
        self.detector = SimplifiedIndustrialDetector(window_size=500, sample_rate=1000)
        self.sample_count = 0
        self.running = False
        self.ser = None
        
        # Simulation parameters
        self.sim_time = 0
        self.sim_failure_mode = 0
    
    def initialize_serial(self):
        """Initialize serial connection"""
        if self.simulate:
            print("🔧 Running in simulation mode")
            return True
        
        try:
            print(f"🔌 Connecting to {self.serial_port} at {self.baud_rate} baud...")
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)
            print(f"✓ Connected to {self.serial_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def read_sensor_data(self):
        """Read sensor data"""
        if self.simulate:
            return self._simulate_sensor_data()
        
        try:
            line = self.ser.readline().decode('utf-8').strip()
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    return x, y, z
            return None
        except (ValueError, UnicodeDecodeError, serial.SerialException):
            return None
    
    def _simulate_sensor_data(self):
        """Simulate realistic sensor data"""
        t = self.sim_time / 1000.0  # Convert to seconds
        base_freq = 50  # Hz
        noise = np.random.normal(0, 0.1)
        
        # Change failure pattern every 10 seconds
        failure_cycle = int(t) // 10
        
        if failure_cycle % 4 == 0:  # Normal
            x = np.sin(2 * np.pi * base_freq * t) + noise
            y = np.cos(2 * np.pi * base_freq * t) + noise
            z = 9.8 + noise
            
        elif failure_cycle % 4 == 1:  # Unbalance
            x = 2 * np.sin(2 * np.pi * base_freq * t) + noise
            y = np.cos(2 * np.pi * base_freq * t) + noise
            z = 9.8 + noise
            
        elif failure_cycle % 4 == 2:  # Bearing wear
            x = np.sin(2 * np.pi * base_freq * t) + 0.5 * np.sin(2 * np.pi * 800 * t) + noise
            y = np.cos(2 * np.pi * base_freq * t) + 0.5 * np.sin(2 * np.pi * 800 * t) + noise
            z = 9.8 + noise
            
        else:  # Looseness
            x = np.sin(2 * np.pi * base_freq * t) + 0.3 * np.sin(4 * np.pi * base_freq * t) + noise
            y = np.cos(2 * np.pi * base_freq * t) + 0.3 * np.cos(4 * np.pi * base_freq * t) + noise
            z = 9.8 + noise
        
        self.sim_time += 1
        return x, y, z
    
    def update_display(self):
        """Update console display"""
        status = self.detector.get_current_status()
        
        print("\033[H\033[J", end="")  # Clear screen
        print("=" * 60)
        print("SIMPLIFIED INDUSTRIAL FAILURE DETECTION")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Samples: {self.sample_count}")
        print(f"Mode: {'SIMULATION' if self.simulate else 'LIVE'}")
        print("-" * 60)
        print(f"Status: {status['status']}")
        print(f"Anomaly Score: {status['anomaly_score']:.3f}")
        print(f"Baseline Learned: {'✓' if status['baseline_learned'] else 'Learning...'}")
        
        if self.detector.is_baseline_learned and len(self.detector.history['lyapunov']) > 0:
            latest_lyapunov = self.detector.history['lyapunov'][-1]
            latest_corr = self.detector.history['correlation_dim'][-1]
            latest_entropy = self.detector.history['entropy'][-1]
            
            print(f"\nLatest Chaos Metrics:")
            print(f"  Lyapunov: {latest_lyapunov:.4f}")
            print(f"  Correlation Dim: {latest_corr:.4f}")
            print(f"  Entropy: {latest_entropy:.4f}")
        
        print("\nPress Ctrl+C to stop")
    
    def run_monitoring(self):
        """Main monitoring loop"""
        print("🚀 Starting Simplified Industrial Failure Detection...")
        
        if not self.initialize_serial():
            return
        
        print("📊 Starting monitoring loop...")
        self.running = True
        
        try:
            while self.running:
                data = self.read_sensor_data()
                
                if data is not None:
                    x, y, z = data
                    self.detector.add_accelerometer_sample(x, y, z)
                    self.sample_count += 1
                    
                    # Update display every 100 samples
                    if self.sample_count % 100 == 0:
                        self.update_display()
                    
                    # Export every 5000 samples
                    if self.sample_count % 5000 == 0 and self.detector.is_baseline_learned:
                        self.detector.export_results()
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        self.running = False
        
        if self.ser is not None:
            self.ser.close()
            print("📡 Serial connection closed")
        
        if self.detector.is_baseline_learned:
            self.detector.export_results()
        
        print("✅ Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Simplified Industrial Machine Failure Detection")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    
    args = parser.parse_args()
    
    monitor = SimplifiedIndustrialMonitor(
        simulate=args.simulate,
        serial_port=args.port,
        baud_rate=args.baud
    )
    
    print("=" * 60)
    print("SIMPLIFIED INDUSTRIAL FAILURE DETECTION SYSTEM")
    print("=" * 60)
    print(f"Serial Port: {args.port}")
    print(f"Baud Rate: {args.baud}")
    print(f"Simulation Mode: {args.simulate}")
    print("=" * 60)
    
    monitor.run_monitoring()


if __name__ == "__main__":
    main()