#!/usr/bin/env python3
"""
Convergence Detection Module for ML Training Runs

Implements automatic convergence detection using multiple algorithms:
- Moving average stability
- Gradient-based detection
- Statistical variance analysis
- Plateau detection

Part of Sprint 2: Advanced Metrics & Analytics
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from scipy import stats


@dataclass
class ConvergenceResult:
    """Result of convergence detection analysis."""
    
    is_converged: bool
    convergence_timestep: Optional[int]
    convergence_value: Optional[float]
    confidence: float  # 0.0 to 1.0
    method: str
    details: Dict
    
    def __str__(self):
        if self.is_converged:
            return (f"Converged at timestep {self.convergence_timestep} "
                   f"(value: {self.convergence_value:.4f}, "
                   f"confidence: {self.confidence:.2%}, method: {self.method})")
        return f"Not converged (confidence: {self.confidence:.2%})"


class ConvergenceDetector:
    """
    Detects convergence in ML training runs using multiple algorithms.
    
    Supports multiple detection methods:
    - Moving average stability
    - Gradient-based detection
    - Statistical variance analysis
    - Plateau detection
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 stability_threshold: float = 0.01,
                 min_samples: int = 200):
        """
        Initialize convergence detector.
        
        Args:
            window_size: Size of moving window for analysis
            stability_threshold: Threshold for stability detection (relative change)
            min_samples: Minimum number of samples required for detection
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)
    
    def detect_convergence(self, 
                          timesteps: List[int], 
                          values: List[float],
                          method: str = 'auto') -> ConvergenceResult:
        """
        Detect convergence in a time series.
        
        Args:
            timesteps: List of timestep values
            values: List of metric values
            method: Detection method ('auto', 'moving_avg', 'gradient', 'variance', 'plateau')
            
        Returns:
            ConvergenceResult object with detection results
        """
        if len(values) < self.min_samples:
            return ConvergenceResult(
                is_converged=False,
                convergence_timestep=None,
                convergence_value=None,
                confidence=0.0,
                method=method,
                details={'reason': 'Insufficient samples', 'samples': len(values)}
            )
        
        # Convert to numpy arrays
        timesteps_arr = np.array(timesteps)
        values_arr = np.array(values)
        
        # Select detection method
        if method == 'auto':
            # Try all methods and use the one with highest confidence
            results = []
            for m in ['moving_avg', 'gradient', 'variance', 'plateau']:
                try:
                    result = self._detect_by_method(timesteps_arr, values_arr, m)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Method {m} failed: {e}")
            
            if not results:
                return ConvergenceResult(
                    is_converged=False,
                    convergence_timestep=None,
                    convergence_value=None,
                    confidence=0.0,
                    method='auto',
                    details={'reason': 'All methods failed'}
                )
            
            # Return result with highest confidence
            return max(results, key=lambda r: r.confidence)
        else:
            return self._detect_by_method(timesteps_arr, values_arr, method)
    
    def _detect_by_method(self, 
                         timesteps: np.ndarray, 
                         values: np.ndarray, 
                         method: str) -> ConvergenceResult:
        """Detect convergence using specific method."""
        if method == 'moving_avg':
            return self._detect_moving_avg(timesteps, values)
        elif method == 'gradient':
            return self._detect_gradient(timesteps, values)
        elif method == 'variance':
            return self._detect_variance(timesteps, values)
        elif method == 'plateau':
            return self._detect_plateau(timesteps, values)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_moving_avg(self, 
                          timesteps: np.ndarray, 
                          values: np.ndarray) -> ConvergenceResult:
        """
        Detect convergence using moving average stability.
        
        Convergence is detected when the moving average remains stable
        (relative change below threshold) for a sustained period.
        """
        # Calculate moving average
        moving_avg = np.convolve(values, np.ones(self.window_size)/self.window_size, mode='valid')
        
        # Calculate relative changes in moving average
        changes = np.abs(np.diff(moving_avg) / (moving_avg[:-1] + 1e-10))
        
        # Find first point where changes remain below threshold
        stable_count = 0
        convergence_idx = None
        
        for i, change in enumerate(changes):
            if change < self.stability_threshold:
                stable_count += 1
                if stable_count >= self.window_size // 2:
                    convergence_idx = i + self.window_size
                    break
            else:
                stable_count = 0
        
        if convergence_idx is not None:
            # Calculate confidence based on stability duration
            remaining_changes = changes[convergence_idx - self.window_size:]
            confidence = 1.0 - np.mean(remaining_changes) / self.stability_threshold
            confidence = np.clip(confidence, 0.0, 1.0)
            
            return ConvergenceResult(
                is_converged=True,
                convergence_timestep=int(timesteps[convergence_idx]),
                convergence_value=float(values[convergence_idx]),
                confidence=float(confidence),
                method='moving_avg',
                details={
                    'stable_duration': len(remaining_changes),
                    'mean_change': float(np.mean(remaining_changes))
                }
            )
        
        return ConvergenceResult(
            is_converged=False,
            convergence_timestep=None,
            convergence_value=None,
            confidence=0.0,
            method='moving_avg',
            details={'reason': 'No stable period found'}
        )
    
    def _detect_gradient(self, 
                        timesteps: np.ndarray, 
                        values: np.ndarray) -> ConvergenceResult:
        """
        Detect convergence using gradient analysis.
        
        Convergence is detected when the gradient approaches zero.
        """
        # Calculate gradient
        gradient = np.gradient(values)
        
        # Calculate moving average of absolute gradient
        abs_gradient = np.abs(gradient)
        moving_grad = np.convolve(abs_gradient, np.ones(self.window_size)/self.window_size, mode='valid')
        
        # Find where gradient becomes very small
        threshold = np.std(gradient) * self.stability_threshold
        
        for i, grad in enumerate(moving_grad):
            if grad < threshold:
                convergence_idx = i + self.window_size
                
                # Calculate confidence
                remaining_grads = moving_grad[i:]
                confidence = 1.0 - np.mean(remaining_grads) / (threshold + 1e-10)
                confidence = np.clip(confidence, 0.0, 1.0)
                
                return ConvergenceResult(
                    is_converged=True,
                    convergence_timestep=int(timesteps[convergence_idx]),
                    convergence_value=float(values[convergence_idx]),
                    confidence=float(confidence),
                    method='gradient',
                    details={
                        'gradient_threshold': float(threshold),
                        'mean_gradient': float(np.mean(remaining_grads))
                    }
                )
        
        return ConvergenceResult(
            is_converged=False,
            convergence_timestep=None,
            convergence_value=None,
            confidence=0.0,
            method='gradient',
            details={'reason': 'Gradient never stabilized'}
        )
    
    def _detect_variance(self, 
                        timesteps: np.ndarray, 
                        values: np.ndarray) -> ConvergenceResult:
        """
        Detect convergence using variance analysis.
        
        Convergence is detected when variance in a moving window becomes small.
        """
        # Calculate rolling variance
        variances = []
        for i in range(len(values) - self.window_size):
            window = values[i:i + self.window_size]
            variances.append(np.var(window))
        
        variances = np.array(variances)
        
        # Find where variance becomes small
        threshold = np.percentile(variances, 10)  # Bottom 10% of variances
        
        for i, var in enumerate(variances):
            if var < threshold:
                convergence_idx = i + self.window_size
                
                # Calculate confidence
                remaining_vars = variances[i:]
                confidence = 1.0 - np.mean(remaining_vars) / (threshold + 1e-10)
                confidence = np.clip(confidence, 0.0, 1.0)
                
                return ConvergenceResult(
                    is_converged=True,
                    convergence_timestep=int(timesteps[convergence_idx]),
                    convergence_value=float(values[convergence_idx]),
                    confidence=float(confidence),
                    method='variance',
                    details={
                        'variance_threshold': float(threshold),
                        'mean_variance': float(np.mean(remaining_vars))
                    }
                )
        
        return ConvergenceResult(
            is_converged=False,
            convergence_timestep=None,
            convergence_value=None,
            confidence=0.0,
            method='variance',
            details={'reason': 'Variance never stabilized'}
        )
    
    def _detect_plateau(self, 
                       timesteps: np.ndarray, 
                       values: np.ndarray) -> ConvergenceResult:
        """
        Detect convergence using plateau detection.
        
        Convergence is detected when values remain within a narrow band.
        """
        # Calculate rolling min/max
        plateau_ranges = []
        for i in range(len(values) - self.window_size):
            window = values[i:i + self.window_size]
            range_val = np.max(window) - np.min(window)
            plateau_ranges.append(range_val)
        
        plateau_ranges = np.array(plateau_ranges)
        
        # Find where range becomes small (plateau)
        mean_val = np.mean(values)
        threshold = mean_val * self.stability_threshold
        
        for i, range_val in enumerate(plateau_ranges):
            if range_val < threshold:
                convergence_idx = i + self.window_size
                
                # Calculate confidence
                remaining_ranges = plateau_ranges[i:]
                confidence = 1.0 - np.mean(remaining_ranges) / (threshold + 1e-10)
                confidence = np.clip(confidence, 0.0, 1.0)
                
                return ConvergenceResult(
                    is_converged=True,
                    convergence_timestep=int(timesteps[convergence_idx]),
                    convergence_value=float(values[convergence_idx]),
                    confidence=float(confidence),
                    method='plateau',
                    details={
                        'plateau_threshold': float(threshold),
                        'mean_range': float(np.mean(remaining_ranges))
                    }
                )
        
        return ConvergenceResult(
            is_converged=False,
            convergence_timestep=None,
            convergence_value=None,
            confidence=0.0,
            method='plateau',
            details={'reason': 'No plateau detected'}
        )

