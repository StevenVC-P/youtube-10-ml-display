#!/usr/bin/env python3
"""
Sample Efficiency Analysis Module for ML Training Runs

Analyzes how efficiently an ML agent learns from samples:
- Learning rate per timestep
- Area under curve (AUC) analysis
- Time to threshold
- Sample efficiency comparison
- Regret analysis

Part of Sprint 2: Advanced Metrics & Analytics
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import integrate


@dataclass
class SampleEfficiencyMetrics:
    """Sample efficiency metrics for a training run."""
    
    auc: float  # Area under curve
    normalized_auc: float  # AUC normalized by max possible
    learning_rate: float  # Average learning rate per timestep
    time_to_threshold: Optional[int]  # Timesteps to reach threshold
    final_performance: float
    initial_performance: float
    total_improvement: float
    efficiency_score: float  # 0-100 composite score
    
    def __str__(self):
        return (f"Sample Efficiency Metrics:\n"
               f"  AUC: {self.auc:.2f} (normalized: {self.normalized_auc:.2%})\n"
               f"  Learning Rate: {self.learning_rate:.6f} per timestep\n"
               f"  Time to Threshold: {self.time_to_threshold or 'N/A'}\n"
               f"  Improvement: {self.initial_performance:.2f} → {self.final_performance:.2f} "
               f"(+{self.total_improvement:.2f})\n"
               f"  Efficiency Score: {self.efficiency_score:.1f}/100")


@dataclass
class ComparisonMetrics:
    """Comparison of sample efficiency between runs."""
    
    run1_efficiency: float
    run2_efficiency: float
    relative_efficiency: float  # run1 / run2
    auc_difference: float
    time_to_threshold_diff: Optional[int]
    winner: str  # 'run1', 'run2', or 'tie'
    
    def __str__(self):
        return (f"Sample Efficiency Comparison:\n"
               f"  Run 1 Efficiency: {self.run1_efficiency:.1f}/100\n"
               f"  Run 2 Efficiency: {self.run2_efficiency:.1f}/100\n"
               f"  Relative Efficiency: {self.relative_efficiency:.2f}x\n"
               f"  AUC Difference: {self.auc_difference:.2f}\n"
               f"  Winner: {self.winner}")


class SampleEfficiencyAnalyzer:
    """
    Analyzes sample efficiency of ML training runs.
    
    Provides methods for:
    - Computing efficiency metrics
    - Comparing runs
    - Identifying optimal learning strategies
    """
    
    def __init__(self, performance_threshold: Optional[float] = None):
        """
        Initialize sample efficiency analyzer.
        
        Args:
            performance_threshold: Threshold for "time to threshold" metric
        """
        self.performance_threshold = performance_threshold
        self.logger = logging.getLogger(__name__)
    
    def analyze_run(self, 
                   timesteps: List[int], 
                   values: List[float],
                   max_possible_value: Optional[float] = None) -> SampleEfficiencyMetrics:
        """
        Analyze sample efficiency of a single run.
        
        Args:
            timesteps: List of timestep values
            values: List of performance values
            max_possible_value: Maximum possible performance (for normalization)
            
        Returns:
            SampleEfficiencyMetrics object
        """
        timesteps_arr = np.array(timesteps)
        values_arr = np.array(values)
        
        # Compute AUC
        auc = self._compute_auc(timesteps_arr, values_arr)
        
        # Normalize AUC if max value provided
        if max_possible_value is not None:
            max_auc = max_possible_value * (timesteps_arr[-1] - timesteps_arr[0])
            normalized_auc = auc / max_auc if max_auc > 0 else 0.0
        else:
            # Normalize by actual max value
            max_auc = np.max(values_arr) * (timesteps_arr[-1] - timesteps_arr[0])
            normalized_auc = auc / max_auc if max_auc > 0 else 0.0
        
        # Compute learning rate
        learning_rate = self._compute_learning_rate(timesteps_arr, values_arr)
        
        # Time to threshold
        time_to_threshold = self._compute_time_to_threshold(
            timesteps_arr, values_arr, self.performance_threshold
        )
        
        # Performance metrics
        initial_performance = float(values_arr[0])
        final_performance = float(values_arr[-1])
        total_improvement = final_performance - initial_performance
        
        # Composite efficiency score (0-100)
        efficiency_score = self._compute_efficiency_score(
            normalized_auc, learning_rate, time_to_threshold, 
            timesteps_arr, total_improvement
        )
        
        return SampleEfficiencyMetrics(
            auc=float(auc),
            normalized_auc=float(normalized_auc),
            learning_rate=float(learning_rate),
            time_to_threshold=time_to_threshold,
            final_performance=final_performance,
            initial_performance=initial_performance,
            total_improvement=total_improvement,
            efficiency_score=float(efficiency_score)
        )
    
    def _compute_auc(self, timesteps: np.ndarray, values: np.ndarray) -> float:
        """Compute area under curve using trapezoidal rule."""
        return float(integrate.trapezoid(values, timesteps))
    
    def _compute_learning_rate(self, timesteps: np.ndarray, values: np.ndarray) -> float:
        """Compute average learning rate per timestep."""
        if len(values) < 2:
            return 0.0
        
        # Use linear regression to estimate learning rate
        coeffs = np.polyfit(timesteps, values, 1)
        learning_rate = coeffs[0]  # Slope
        
        return float(learning_rate)
    
    def _compute_time_to_threshold(self, 
                                   timesteps: np.ndarray, 
                                   values: np.ndarray,
                                   threshold: Optional[float]) -> Optional[int]:
        """Compute timesteps required to reach threshold."""
        if threshold is None:
            return None
        
        # Find first timestep where value >= threshold
        indices = np.where(values >= threshold)[0]
        
        if len(indices) > 0:
            return int(timesteps[indices[0]])
        
        return None  # Threshold never reached
    
    def _compute_efficiency_score(self, 
                                 normalized_auc: float,
                                 learning_rate: float,
                                 time_to_threshold: Optional[int],
                                 timesteps: np.ndarray,
                                 total_improvement: float) -> float:
        """
        Compute composite efficiency score (0-100).
        
        Combines multiple factors:
        - Normalized AUC (40%)
        - Learning rate (30%)
        - Time to threshold (20%)
        - Total improvement (10%)
        """
        score = 0.0
        
        # AUC component (40 points)
        score += normalized_auc * 40
        
        # Learning rate component (30 points)
        # Normalize learning rate (higher is better)
        if total_improvement > 0:
            total_time = timesteps[-1] - timesteps[0]
            max_learning_rate = total_improvement / total_time
            normalized_lr = min(learning_rate / max_learning_rate, 1.0) if max_learning_rate > 0 else 0.0
            score += normalized_lr * 30
        
        # Time to threshold component (20 points)
        if time_to_threshold is not None:
            total_time = timesteps[-1] - timesteps[0]
            # Earlier is better
            time_score = 1.0 - (time_to_threshold - timesteps[0]) / total_time
            score += max(time_score, 0.0) * 20
        else:
            # Penalty for not reaching threshold
            score += 0
        
        # Total improvement component (10 points)
        # Normalize by initial value
        if abs(timesteps[0]) > 1e-10:
            improvement_ratio = min(abs(total_improvement / timesteps[0]), 1.0)
            score += improvement_ratio * 10
        
        return min(max(score, 0.0), 100.0)
    
    def compare_runs(self, 
                    run1_timesteps: List[int],
                    run1_values: List[float],
                    run2_timesteps: List[int],
                    run2_values: List[float]) -> ComparisonMetrics:
        """
        Compare sample efficiency of two runs.
        
        Args:
            run1_timesteps: Timesteps for run 1
            run1_values: Values for run 1
            run2_timesteps: Timesteps for run 2
            run2_values: Values for run 2
            
        Returns:
            ComparisonMetrics object
        """
        # Analyze both runs
        metrics1 = self.analyze_run(run1_timesteps, run1_values)
        metrics2 = self.analyze_run(run2_timesteps, run2_values)
        
        # Compute relative efficiency
        relative_efficiency = (metrics1.efficiency_score / metrics2.efficiency_score 
                              if metrics2.efficiency_score > 0 else float('inf'))
        
        # AUC difference
        auc_diff = metrics1.auc - metrics2.auc
        
        # Time to threshold difference
        if metrics1.time_to_threshold is not None and metrics2.time_to_threshold is not None:
            time_diff = metrics1.time_to_threshold - metrics2.time_to_threshold
        else:
            time_diff = None
        
        # Determine winner
        if abs(metrics1.efficiency_score - metrics2.efficiency_score) < 1.0:
            winner = 'tie'
        elif metrics1.efficiency_score > metrics2.efficiency_score:
            winner = 'run1'
        else:
            winner = 'run2'
        
        return ComparisonMetrics(
            run1_efficiency=metrics1.efficiency_score,
            run2_efficiency=metrics2.efficiency_score,
            relative_efficiency=float(relative_efficiency),
            auc_difference=float(auc_diff),
            time_to_threshold_diff=time_diff,
            winner=winner
        )
    
    def compute_regret(self, 
                      timesteps: List[int],
                      values: List[float],
                      optimal_value: float) -> Dict:
        """
        Compute regret analysis.
        
        Regret = difference between optimal performance and actual performance
        
        Args:
            timesteps: List of timesteps
            values: List of performance values
            optimal_value: Optimal/maximum possible performance
            
        Returns:
            Dictionary with regret metrics
        """
        timesteps_arr = np.array(timesteps)
        values_arr = np.array(values)
        
        # Instantaneous regret at each timestep
        regret = optimal_value - values_arr
        
        # Cumulative regret
        cumulative_regret = np.cumsum(regret)
        
        # Average regret
        avg_regret = np.mean(regret)
        
        # Final regret
        final_regret = regret[-1]
        
        # Regret reduction rate
        if len(regret) > 1:
            regret_reduction = (regret[0] - regret[-1]) / (timesteps_arr[-1] - timesteps_arr[0])
        else:
            regret_reduction = 0.0
        
        return {
            'instantaneous_regret': regret.tolist(),
            'cumulative_regret': cumulative_regret.tolist(),
            'average_regret': float(avg_regret),
            'final_regret': float(final_regret),
            'regret_reduction_rate': float(regret_reduction),
            'total_cumulative_regret': float(cumulative_regret[-1])
        }
    
    def identify_learning_phases(self, 
                                timesteps: List[int],
                                values: List[float],
                                n_phases: int = 3) -> List[Dict]:
        """
        Identify distinct learning phases in training.
        
        Args:
            timesteps: List of timesteps
            values: List of performance values
            n_phases: Number of phases to identify
            
        Returns:
            List of phase dictionaries
        """
        timesteps_arr = np.array(timesteps)
        values_arr = np.array(values)
        
        # Split into phases
        phase_size = len(values_arr) // n_phases
        phases = []
        
        for i in range(n_phases):
            start_idx = i * phase_size
            end_idx = (i + 1) * phase_size if i < n_phases - 1 else len(values_arr)
            
            phase_timesteps = timesteps_arr[start_idx:end_idx]
            phase_values = values_arr[start_idx:end_idx]
            
            # Compute phase metrics
            phase_learning_rate = self._compute_learning_rate(phase_timesteps, phase_values)
            phase_improvement = phase_values[-1] - phase_values[0]
            
            phases.append({
                'phase_number': i + 1,
                'start_timestep': int(phase_timesteps[0]),
                'end_timestep': int(phase_timesteps[-1]),
                'start_value': float(phase_values[0]),
                'end_value': float(phase_values[-1]),
                'improvement': float(phase_improvement),
                'learning_rate': float(phase_learning_rate),
                'duration': int(phase_timesteps[-1] - phase_timesteps[0])
            })
        
        return phases

