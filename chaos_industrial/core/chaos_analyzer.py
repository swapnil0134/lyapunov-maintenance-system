"""
Optimized Chaos Theory Metrics for Industrial Machine Failure Detection

This module provides fast, efficient implementations of chaos theory metrics
specifically optimized for real-time industrial machinery monitoring.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class FastChaosAnalyzer:
    """Optimized chaos metrics calculation for real-time industrial monitoring"""
    
    def __init__(self, cache_size=100):
        self.cache_size = cache_size
        self.embedding_cache = deque(maxlen=cache_size)
        self.distance_cache = {}
        
    def fast_lyapunov(self, data, emb_dim=3, delay=1, min_tsep=10):
        """
        Fast Lyapunov exponent estimation using vectorized operations
        
        Args:
            data: Time series data
            emb_dim: Embedding dimension
            delay: Time delay
            min_tsep: Minimum time separation for neighbors
            
        Returns:
            Largest Lyapunov exponent
        """
        n = len(data)
        if n < (emb_dim - 1) * delay + min_tsep + 10:
            return 0.0
        
        # Create embedding efficiently
        N = n - (emb_dim - 1) * delay
        embedded = np.zeros((N, emb_dim))
        for i in range(emb_dim):
            embedded[:, i] = data[i * delay : i * delay + N]
        
        # Vectorized distance calculation
        distances = np.linalg.norm(embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :], axis=2)
        
        # Find nearest neighbors (excluding temporal neighbors)
        np.fill_diagonal(distances, np.inf)
        for i in range(N):
            start = max(0, i - min_tsep)
            end = min(N, i + min_tsep + 1)
            distances[i, start:end] = np.inf
        
        # Track divergence for multiple time steps
        divergence_data = []
        for i in range(N - min_tsep - 10):
            nearest_idx = np.argmin(distances[i])
            if distances[i, nearest_idx] < np.inf:
                # Track divergence for next few steps
                max_steps = min(20, N - max(i, nearest_idx))
                for k in range(1, max_steps):
                    if i + k < N and nearest_idx + k < N:
                        d_ik = np.linalg.norm(embedded[i + k] - embedded[nearest_idx + k])
                        if d_ik > 0:
                            divergence_data.append((k, np.log(d_ik)))
        
        if len(divergence_data) < 10:
            return 0.0
        
        # Linear fit for Lyapunov exponent
        divergence_data = np.array(divergence_data)
        k_vals = divergence_data[:, 0]
        log_dist = divergence_data[:, 1]
        
        # Average by time step
        unique_k = np.unique(k_vals)
        avg_log_dist = []
        for k in unique_k:
            if k <= 10:  # Limit to short-term divergence
                avg_log_dist.append(np.mean(log_dist[k_vals == k]))
        
        if len(avg_log_dist) < 2:
            return 0.0
        
        # Linear regression
        coeffs = np.polyfit(unique_k[:len(avg_log_dist)], avg_log_dist, 1)
        return max(0.0, coeffs[0])  # Ensure non-negative
    
    def fast_correlation_dimension(self, data, emb_dim=5, delay=1, max_points=200):
        """
        Fast correlation dimension estimation using sampling
        
        Args:
            data: Time series data
            emb_dim: Embedding dimension
            delay: Time delay
            max_points: Maximum points for distance calculation (speed optimization)
            
        Returns:
            Correlation dimension estimate
        """
        n = len(data)
        N = n - (emb_dim - 1) * delay
        
        if N < 50:
            return 2.0
        
        # Create embedding
        embedded = np.zeros((N, emb_dim))
        for i in range(emb_dim):
            embedded[:, i] = data[i * delay : i * delay + N]
        
        # Sample points for speed
        sample_size = min(N, max_points)
        sample_indices = np.random.choice(N, sample_size, replace=False)
        sample_embedded = embedded[sample_indices]
        
        # Calculate pairwise distances efficiently
        if sample_size > 1:
            distances = pdist(sample_embedded, 'euclidean')
        else:
            return 2.0
        
        if len(distances) == 0:
            return 2.0
        
        # Calculate correlation sum for different radii
        percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        radii = np.percentile(distances, percentiles)
        
        # Remove zero radius
        radii = radii[radii > 0]
        if len(radii) < 3:
            return 2.0
        
        corr_sums = []
        for r in radii:
            corr_sum = np.sum(distances < r) / len(distances)
            if corr_sum > 0:
                corr_sums.append(corr_sum)
        
        if len(corr_sums) < 2:
            return 2.0
        
        # Estimate dimension from slope
        log_r = np.log(radii[:len(corr_sums)])
        log_c = np.log(corr_sums)
        
        # Linear fit on middle range
        mid_start = len(log_r) // 4
        mid_end = 3 * len(log_r) // 4
        
        if mid_end - mid_start < 2:
            return 2.0
        
        coeffs = np.polyfit(log_r[mid_start:mid_end], log_c[mid_start:mid_end], 1)
        correlation_dim = max(0.5, min(10.0, coeffs[0]))
        
        return correlation_dim
    
    def fast_approximate_entropy(self, data, m=2, r_factor=0.2):
        """
        Fast Approximate Entropy calculation
        
        Args:
            data: Time series data
            m: Pattern length
            r_factor: Tolerance factor
            
        Returns:
            Approximate entropy value
        """
        N = len(data)
        if N < m + 1:
            return 0.0
        
        r = r_factor * np.std(data)
        
        def _phi(m_val):
            if N - m_val + 1 < 1:
                return 0.0
            
            # Create patterns efficiently
            patterns = np.zeros((N - m_val + 1, m_val))
            for i in range(m_val):
                patterns[:, i] = data[i:i + N - m_val + 1]
            
            # Calculate similarities
            C = np.zeros(N - m_val + 1)
            for i in range(N - m_val + 1):
                # Vectorized distance calculation
                diffs = np.abs(patterns - patterns[i])
                max_diffs = np.max(diffs, axis=1)
                C[i] = np.sum(max_diffs <= r)
            
            C = C / (N - m_val + 1)
            
            # Avoid log(0)
            C = np.maximum(C, 1e-10)
            return np.mean(np.log(C))
        
        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            apen = phi_m - phi_m1
            return max(0.0, apen)
        except:
            return 0.0
    
    def calculate_all_metrics(self, data, emb_dim=3, delay=1):
        """
        Calculate all chaos metrics efficiently
        
        Args:
            data: Time series data
            emb_dim: Embedding dimension
            delay: Time delay
            
        Returns:
            Dictionary with all chaos metrics
        """
        if len(data) < 100:
            return {
                'lyapunov': 0.0,
                'correlation_dimension': 2.0,
                'approximate_entropy': 0.0
            }
        
        # Parallel calculation would be ideal, but sequential for reliability
        lyapunov = self.fast_lyapunov(data, emb_dim, delay)
        corr_dim = self.fast_correlation_dimension(data, emb_dim + 2, delay)
        apen = self.fast_approximate_entropy(data, m=2)
        
        return {
            'lyapunov': lyapunov,
            'correlation_dimension': corr_dim,
            'approximate_entropy': apen
        }


class ChaosMetricsValidator:
    """Validates and filters chaos metrics for reliability"""
    
    def __init__(self):
        self.metric_ranges = {
            'lyapunov': (0.0, 2.0),
            'correlation_dimension': (0.5, 10.0),
            'approximate_entropy': (0.0, 2.0)
        }
    
    def validate_metrics(self, metrics):
        """
        Validate and filter chaos metrics
        
        Args:
            metrics: Dictionary of chaos metrics
            
        Returns:
            Validated metrics dictionary
        """
        validated = {}
        for metric, value in metrics.items():
            if metric in self.metric_ranges:
                min_val, max_val = self.metric_ranges[metric]
                validated[metric] = max(min_val, min(max_val, value))
            else:
                validated[metric] = value
        
        return validated
    
    def is_reliable(self, metrics, data_length):
        """
        Check if metrics are reliable based on data characteristics
        
        Args:
            metrics: Chaos metrics dictionary
            data_length: Length of input data
            
        Returns:
            Boolean indicating reliability
        """
        # Minimum data length requirement
        if data_length < 200:
            return False
        
        # Check for extreme values that indicate calculation problems
        if metrics['lyapunov'] > 1.5 or metrics['lyapunov'] < 0.0:
            return False
        
        if metrics['correlation_dimension'] > 8.0 or metrics['correlation_dimension'] < 0.5:
            return False
        
        if metrics['approximate_entropy'] > 1.5:
            return False
        
        return True