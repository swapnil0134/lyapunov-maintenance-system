#!/usr/bin/env python3
"""
Test script for Industrial Machine Failure Detection System

This script tests the basic functionality of the industrial failure detection
system using simulated data to ensure all components work correctly.
"""

import sys
import numpy as np
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, '/home/swapnil/anaconda_projects/chaos')

try:
    from chaos_industrial.detection import IndustrialFailureDetector
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check that all modules are properly installed")
    sys.exit(1)


def test_industrial_detector():
    """Test the industrial failure detection system"""
    print("🧪 Testing Industrial Machine Failure Detection System")
    print("=" * 60)
    
    # Create detector with smaller window for testing
    detector = IndustrialFailureDetector(window_size=200, sample_rate=1000)
    
    print("✅ IndustrialFailureDetector created successfully")
    
    # Test with simulated data
    print("\n📊 Testing with simulated failure patterns...")
    
    # Generate test data with different failure patterns
    test_duration = 10  # seconds
    sample_rate = 1000
    samples_per_phase = sample_rate * 2  # 2 seconds per failure pattern
    
    for phase in range(5):
        print(f"\n🔄 Testing phase {phase + 1}/5...")
        
        # Generate data for this phase
        t = np.linspace(0, 2, samples_per_phase)
        base_freq = 50  # Hz
        
        if phase == 0:  # Normal operation
            print("  📈 Normal operation")
            x = np.sin(2 * np.pi * base_freq * t) + np.random.normal(0, 0.1, len(t))
            y = np.cos(2 * np.pi * base_freq * t) + np.random.normal(0, 0.1, len(t))
            z = 9.8 + np.random.normal(0, 0.05, len(t))
            
        elif phase == 1:  # Unbalance
            print("  ⚖️  Unbalance pattern")
            x = 2 * np.sin(2 * np.pi * base_freq * t) + np.random.normal(0, 0.1, len(t))
            y = np.cos(2 * np.pi * base_freq * t) + np.random.normal(0, 0.1, len(t))
            z = 9.8 + np.random.normal(0, 0.05, len(t))
            
        elif phase == 2:  # Bearing wear
            print("  🛢️  Bearing wear pattern")
            x = np.sin(2 * np.pi * base_freq * t) + 0.5 * np.sin(2 * np.pi * 800 * t) + np.random.normal(0, 0.1, len(t))
            y = np.cos(2 * np.pi * base_freq * t) + 0.5 * np.sin(2 * np.pi * 800 * t) + np.random.normal(0, 0.1, len(t))
            z = 9.8 + np.random.normal(0, 0.05, len(t))
            
        elif phase == 3:  # Misalignment
            print("  📐 Misalignment pattern")
            x = np.sin(2 * np.pi * base_freq * t) + 0.3 * np.sin(4 * np.pi * base_freq * t) + np.random.normal(0, 0.1, len(t))
            y = np.cos(2 * np.pi * base_freq * t) + 0.3 * np.cos(4 * np.pi * base_freq * t) + np.random.normal(0, 0.1, len(t))
            z = 9.8 + 0.2 * np.sin(2 * np.pi * base_freq * t) + np.random.normal(0, 0.05, len(t))
            
        else:  # Looseness
            print("  🔩 Looseness pattern")
            x = np.sin(2 * np.pi * base_freq * t) + 0.3 * np.sin(4 * np.pi * base_freq * t) + 0.2 * np.sin(6 * np.pi * base_freq * t) + np.random.normal(0, 0.1, len(t))
            y = np.cos(2 * np.pi * base_freq * t) + 0.3 * np.cos(4 * np.pi * base_freq * t) + 0.2 * np.cos(6 * np.pi * base_freq * t) + np.random.normal(0, 0.1, len(t))
            z = 9.8 + np.random.normal(0, 0.05, len(t))
        
        # Feed data to detector
        for i in range(len(x)):
            detector.add_accelerometer_sample(x[i], y[i], z[i])
            
            # Check status periodically
            if i % 500 == 0 and i > 0:
                status = detector.get_current_status()
                print(f"    Sample {i}/{len(x)}: {status['status'][:50]}...")
        
        # Wait a bit between phases
        time.sleep(0.5)
    
    # Get final status
    print("\n📋 Final System Status:")
    final_status = detector.get_current_status()
    print(f"  Status: {final_status['status']}")
    print(f"  Anomaly Score: {final_status['anomaly_score']:.3f}")
    print(f"  Baseline Learned: {final_status['baseline_learned']}")
    
    if final_status['failure_modes']:
        print("  Failure Modes Detected:")
        for mode, info in final_status['failure_modes'].items():
            if info.get('detected', False):
                severity = info.get('severity', 'unknown')
                confidence = info.get('confidence', 0) * 100
                print(f"    ⚠️  {mode.replace('_', ' ').title()}: {severity} ({confidence:.1f}% confidence)")
    
    # Export results
    print("\n💾 Exporting test results...")
    detector.export_results('industrial_failure_test.csv')
    
    print("\n✅ Test completed successfully!")
    print("📁 Results saved to 'industrial_failure_test.csv'")
    
    return True


def test_individual_components():
    """Test individual components"""
    print("\n🔧 Testing Individual Components")
    print("-" * 40)
    
    try:
        # Test chaos analyzer
        from chaos_industrial.core import FastChaosAnalyzer
        chaos_analyzer = FastChaosAnalyzer()
        test_data = np.random.normal(0, 1, 1000)
        metrics = chaos_analyzer.calculate_all_metrics(test_data)
        print(f"✅ Chaos Analyzer: {len(metrics)} metrics calculated")
        
        # Test frequency analyzer
        from chaos_industrial.core import FrequencyAnalyzer
        freq_analyzer = FrequencyAnalyzer()
        freq_features = freq_analyzer.extract_frequency_features(test_data)
        print(f"✅ Frequency Analyzer: {len(freq_features)} features extracted")
        
        # Test sensor processor
        from chaos_industrial.core import SensorProcessor
        sensor_proc = SensorProcessor()
        sensor_proc.add_sensor_data('accelerometer', test_data)
        fused_data = sensor_proc.fuse_sensor_data()
        print(f"✅ Sensor Processor: {len(fused_data)} samples fused")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Industrial Machine Failure Detection System - Test Suite")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test individual components
    if not test_individual_components():
        print("\n❌ Component tests failed. Exiting.")
        return False
    
    # Test main detector
    if not test_industrial_detector():
        print("\n❌ Main detector test failed. Exiting.")
        return False
    
    print(f"\n✅ All tests passed successfully!")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)