#!/usr/bin/env python3
"""
Industrial Machine Failure Detection System - Main Application

Real-time monitoring system for industrial machinery using chaos theory,
frequency analysis, and multi-sensor data fusion for comprehensive 
failure detection and prevention.

Usage:
    python industrial_failure_monitor.py --port /dev/ttyUSB0 --baud 115200
    python industrial_failure_monitor.py --config config.yaml
    python industrial_failure_monitor.py --simulate
"""

import sys
import time
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import serial
import warnings
warnings.filterwarnings('ignore')

# Import our industrial failure detection system
try:
    from chaos_industrial.detection import IndustrialFailureDetector
except ImportError:
    print("Error: Could not import industrial failure detection modules")
    print("Please ensure all modules are properly installed")
    sys.exit(1)


class IndustrialFailureMonitor:
    """Main monitoring application for industrial machine failure detection"""
    
    def __init__(self, config=None):
        self.config = config or self._load_default_config()
        
        # Initialize the industrial failure detector
        self.detector = IndustrialFailureDetector(
            window_size=self.config.get('window_size', 1000),
            sample_rate=self.config.get('sample_rate', 1000),
            config=self.config.get('detection_config', None)
        )
        
        # Monitoring parameters
        self.serial_port = self.config.get('serial_port', '/dev/ttyUSB0')
        self.baud_rate = self.config.get('baud_rate', 115200)
        self.update_interval = self.config.get('update_interval', 50)
        self.export_interval = self.config.get('export_interval', 1000)
        
        # State tracking
        self.sample_count = 0
        self.last_export_time = time.time()
        self.running = False
        self.simulate_mode = self.config.get('simulate', False)
        
        # Serial connection
        self.ser = None
        
        # Plotting
        self.live_plotting = self.config.get('live_plotting', False)
        self.fig = None
        self.axes = None
    
    def _load_default_config(self):
        """Load default configuration"""
        return {
            'window_size': 1000,
            'sample_rate': 1000,
            'serial_port': '/dev/ttyUSB0',
            'baud_rate': 115200,
            'update_interval': 50,
            'export_interval': 1000,
            'live_plotting': False,
            'simulate': False,
            'detection_config': None
        }
    
    def _load_config_from_file(self, config_file):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuration loaded from {config_file}")
            return config
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return self._load_default_config()
    
    def initialize_serial_connection(self):
        """Initialize serial connection to sensor"""
        if self.simulate_mode:
            print("🔧 Running in simulation mode")
            return True
        
        try:
            print(f"🔌 Connecting to {self.serial_port} at {self.baud_rate} baud...")
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Allow connection to stabilize
            print(f"✓ Connected to {self.serial_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to {self.serial_port}: {e}")
            return False
    
    def read_sensor_data(self):
        """Read sensor data from serial port or simulate"""
        if self.simulate_mode:
            return self._simulate_sensor_data()
        
        try:
            line = self.ser.readline().decode('utf-8').strip()
            
            # Parse accelerometer data (format: "x,y,z")
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        return x, y, z
                    except ValueError:
                        return None
            
            return None
            
        except (ValueError, UnicodeDecodeError, serial.SerialException):
            return None
    
    def _simulate_sensor_data(self):
        """Simulate sensor data for testing"""
        # Generate realistic vibration data with occasional failures
        t = time.time()
        base_freq = 50  # Hz
        noise = np.random.normal(0, 0.1)
        
        # Add different failure patterns periodically
        failure_mode = int(t) % 100  # Change failure every 100 seconds
        
        if failure_mode < 25:  # Normal operation
            x = np.sin(2 * np.pi * base_freq * t / 1000) + noise
            y = np.cos(2 * np.pi * base_freq * t / 1000) + noise
            z = 9.8 + noise
            
        elif failure_mode < 50:  # Unbalance
            x = 2 * np.sin(2 * np.pi * base_freq * t / 1000) + noise
            y = np.cos(2 * np.pi * base_freq * t / 1000) + noise
            z = 9.8 + noise
            
        elif failure_mode < 75:  # Bearing wear (high frequency)
            x = np.sin(2 * np.pi * base_freq * t / 1000) + 0.5 * np.sin(2 * np.pi * 800 * t / 1000) + noise
            y = np.cos(2 * np.pi * base_freq * t / 1000) + 0.5 * np.sin(2 * np.pi * 800 * t / 1000) + noise
            z = 9.8 + noise
            
        else:  # Looseness (multiple harmonics)
            x = np.sin(2 * np.pi * base_freq * t / 1000) + 0.3 * np.sin(4 * np.pi * base_freq * t / 1000) + noise
            y = np.cos(2 * np.pi * base_freq * t / 1000) + 0.3 * np.cos(4 * np.pi * base_freq * t / 1000) + noise
            z = 9.8 + noise
        
        return x, y, z
    
    def update_display(self):
        """Update console display with current status"""
        status = self.detector.get_current_status()
        
        # Clear screen (simple approach)
        print("\033[H\033[J", end="")
        
        # Header
        print("=" * 80)
        print("INDUSTRIAL MACHINE FAILURE DETECTION SYSTEM")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Samples Processed: {self.sample_count}")
        print(f"Mode: {'SIMULATION' if self.simulate_mode else 'LIVE'}")
        print("-" * 80)
        
        # Status
        print(f"System Status: {status['status']}")
        print(f"Anomaly Score: {status['anomaly_score']:.3f}")
        print(f"Baseline Learned: {'✓' if status['baseline_learned'] else 'Learning...'}")
        
        # Failure modes
        if status['failure_modes']:
            print("\nFailure Modes Detected:")
            for mode, info in status['failure_modes'].items():
                severity_emoji = {
                    'normal': '✓',
                    'minor': '⚡',
                    'moderate': '⚠️',
                    'severe': '🚨'
                }
                emoji = severity_emoji.get(info.get('severity', 'normal'), '❓')
                detected = "DETECTED" if info.get('detected', False) else "Normal"
                confidence = info.get('confidence', 0) * 100
                print(f"  {emoji} {mode.replace('_', ' ').title()}: {detected} ({confidence:.1f}% confidence)")
        
        # Sensor health
        sensor_health = self.detector.sensor_processor.get_all_sensor_health()
        print(f"\nSensor Health:")
        for sensor, health in sensor_health.items():
            health_status = "✓" if health.get('healthy', False) else "❌"
            print(f"  {health_status} {sensor.title()}: {health.get('error', 'OK')}")
        
        print("\nPress Ctrl+C to stop monitoring")
    
    def setup_live_plotting(self):
        """Setup live plotting window"""
        if not self.live_plotting:
            return
        
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Industrial Machine Failure Detection - Live Monitoring")
        
        # Initialize plots
        self.axes[0, 0].set_title("Anomaly Score")
        self.axes[0, 0].set_ylim([0, 1])
        self.anomaly_line, = self.axes[0, 0].plot([], [], 'b-')
        
        self.axes[0, 1].set_title("Chaos Metrics")
        self.axes[1, 0].set_title("Frequency Spectrum")
        self.axes[1, 1].set_title("Failure Mode Confidence")
        
        plt.tight_layout()
    
    def update_live_plot(self):
        """Update live plotting"""
        if not self.live_plotting or self.fig is None:
            return
        
        if len(self.detector.history['anomaly_score']) == 0:
            return
        
        # Update anomaly score plot
        x_data = range(len(self.detector.history['anomaly_score']))
        self.anomaly_line.set_data(x_data, self.detector.history['anomaly_score'])
        self.axes[0, 0].set_xlim([0, max(100, len(x_data))])
        
        # Update other plots (simplified for now)
        if len(self.detector.history['chaos_metrics']) > 0:
            latest_chaos = self.detector.history['chaos_metrics'][-1]
            metrics = list(latest_chaos.keys())
            values = list(latest_chaos.values())
            self.axes[0, 1].clear()
            self.axes[0, 1].bar(metrics, values)
            self.axes[0, 1].set_title("Latest Chaos Metrics")
            self.axes[0, 1].tick_params(axis='x', rotation=45)
        
        plt.pause(0.01)
    
    def run_monitoring(self):
        """Main monitoring loop"""
        print("🚀 Starting Industrial Machine Failure Detection System...")
        
        if not self.initialize_serial_connection():
            return
        
        self.setup_live_plotting()
        
        print("📊 Starting monitoring loop...")
        self.running = True
        
        try:
            while self.running:
                # Read sensor data
                sensor_data = self.read_sensor_data()
                
                if sensor_data is not None:
                    x, y, z = sensor_data
                    
                    # Add to detector
                    self.detector.add_accelerometer_sample(x, y, z)
                    self.sample_count += 1
                    
                    # Update display
                    if self.sample_count % self.update_interval == 0:
                        self.update_display()
                    
                    # Update live plot
                    if self.sample_count % (self.update_interval // 2) == 0:
                        self.update_live_plot()
                    
                    # Export results periodically
                    current_time = time.time()
                    if current_time - self.last_export_time > self.export_interval:
                        self.detector.export_results()
                        self.last_export_time = current_time
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        self.running = False
        
        # Close serial connection
        if self.ser is not None:
            self.ser.close()
            print("📡 Serial connection closed")
        
        # Export final results
        if self.detector.is_baseline_learned:
            print("💾 Exporting final results...")
            self.detector.export_results()
        
        # Save plots
        if self.live_plotting and self.fig is not None:
            plt.savefig('industrial_failure_monitoring.png', dpi=300, bbox_inches='tight')
            print("📈 Live plot saved to 'industrial_failure_monitoring.png'")
            plt.close()
        
        print("✅ Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Industrial Machine Failure Detection System")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for sensor connection")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate for serial connection")
    parser.add_argument("--config", help="Configuration file path (YAML)")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument("--plot", action="store_true", help="Enable live plotting")
    parser.add_argument("--window", type=int, default=1000, help="Window size for analysis")
    parser.add_argument("--rate", type=int, default=1000, help="Sample rate (Hz)")
    parser.add_argument("--interval", type=int, default=50, help="Display update interval")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'serial_port': args.port,
        'baud_rate': args.baud,
        'window_size': args.window,
        'sample_rate': args.rate,
        'update_interval': args.interval,
        'live_plotting': args.plot,
        'simulate': args.simulate
    }
    
    if args.config:
        file_config = IndustrialFailureMonitor()._load_config_from_file(args.config)
        config.update(file_config)
    
    # Create and run monitor
    monitor = IndustrialFailureMonitor(config)
    
    print("=" * 80)
    print("INDUSTRIAL MACHINE FAILURE DETECTION SYSTEM")
    print("=" * 80)
    print(f"Serial Port: {config['serial_port']}")
    print(f"Baud Rate: {config['baud_rate']}")
    print(f"Window Size: {config['window_size']}")
    print(f"Sample Rate: {config['sample_rate']} Hz")
    print(f"Live Plotting: {'Enabled' if config['live_plotting'] else 'Disabled'}")
    print(f"Simulation Mode: {'Enabled' if config['simulate'] else 'Disabled'}")
    print("=" * 80)
    
    monitor.run_monitoring()


if __name__ == "__main__":
    main()