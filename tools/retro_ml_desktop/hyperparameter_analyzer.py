#!/usr:bin/env python3
"""
Hyperparameter Sensitivity Analysis Module

Analyzes the impact of hyperparameters on training performance:
- Correlation analysis
- Sensitivity ranking
- Interaction effects
- Optimal range identification
- Feature importance

Part of Sprint 2: Advanced Metrics & Analytics
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr


@dataclass
class CorrelationResult:
    """Correlation between hyperparameter and performance."""
    
    hyperparameter: str
    correlation: float
    p_value: float
    is_significant: bool
    method: str
    interpretation: str
    
    def __str__(self):
        sig = "significant" if self.is_significant else "not significant"
        strength = self._interpret_strength()
        return (f"{self.hyperparameter}: r={self.correlation:.3f}, "
               f"p={self.p_value:.4f} ({sig}, {strength})\n"
               f"{self.interpretation}")
    
    def _interpret_strength(self) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(self.correlation)
        if abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.7:
            return "moderate"
        else:
            return "strong"


@dataclass
class SensitivityRanking:
    """Ranking of hyperparameters by sensitivity."""
    
    rankings: List[Tuple[str, float]]  # (hyperparameter, importance_score)
    method: str
    
    def __str__(self):
        lines = [f"Hyperparameter Sensitivity Ranking ({self.method}):"]
        for i, (param, score) in enumerate(self.rankings, 1):
            lines.append(f"  {i}. {param}: {score:.4f}")
        return "\n".join(lines)


@dataclass
class InteractionEffect:
    """Interaction effect between two hyperparameters."""
    
    param1: str
    param2: str
    interaction_strength: float
    is_significant: bool
    interpretation: str
    
    def __str__(self):
        sig = "significant" if self.is_significant else "not significant"
        return (f"Interaction: {self.param1} × {self.param2}\n"
               f"  Strength: {self.interaction_strength:.4f} ({sig})\n"
               f"  {self.interpretation}")


class HyperparameterAnalyzer:
    """
    Analyzes hyperparameter sensitivity and correlations.
    
    Provides methods for:
    - Correlation analysis
    - Sensitivity ranking
    - Interaction detection
    - Optimal range identification
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize hyperparameter analyzer.
        
        Args:
            significance_level: Significance level for statistical tests
        """
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
    
    def analyze_correlation(self, 
                          hyperparameter_values: List[float],
                          performance_values: List[float],
                          hyperparameter_name: str,
                          method: str = 'auto') -> CorrelationResult:
        """
        Analyze correlation between hyperparameter and performance.
        
        Args:
            hyperparameter_values: Values of the hyperparameter
            performance_values: Corresponding performance values
            hyperparameter_name: Name of the hyperparameter
            method: Correlation method ('auto', 'pearson', 'spearman')
            
        Returns:
            CorrelationResult object
        """
        hp_arr = np.array(hyperparameter_values)
        perf_arr = np.array(performance_values)
        
        if method == 'auto':
            # Check for linearity - use Pearson if linear, Spearman otherwise
            # Simple heuristic: if data looks monotonic, use Spearman
            method = 'spearman'  # Default to more robust method
        
        if method == 'pearson':
            correlation, p_value = pearsonr(hp_arr, perf_arr)
            method_name = 'Pearson'
        elif method == 'spearman':
            correlation, p_value = spearmanr(hp_arr, perf_arr)
            method_name = 'Spearman'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        is_significant = p_value < self.significance_level
        
        # Interpretation
        if is_significant:
            if correlation > 0:
                interpretation = f"Higher {hyperparameter_name} is associated with better performance"
            else:
                interpretation = f"Lower {hyperparameter_name} is associated with better performance"
        else:
            interpretation = f"No significant relationship between {hyperparameter_name} and performance"
        
        return CorrelationResult(
            hyperparameter=hyperparameter_name,
            correlation=float(correlation),
            p_value=float(p_value),
            is_significant=is_significant,
            method=method_name,
            interpretation=interpretation
        )
    
    def rank_sensitivity(self, 
                        hyperparameters: Dict[str, List[float]],
                        performance_values: List[float],
                        method: str = 'correlation') -> SensitivityRanking:
        """
        Rank hyperparameters by their sensitivity/importance.
        
        Args:
            hyperparameters: Dictionary mapping hyperparameter names to values
            performance_values: Corresponding performance values
            method: Ranking method ('correlation', 'variance')
            
        Returns:
            SensitivityRanking object
        """
        if method == 'correlation':
            return self._rank_by_correlation(hyperparameters, performance_values)
        elif method == 'variance':
            return self._rank_by_variance(hyperparameters, performance_values)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _rank_by_correlation(self, 
                            hyperparameters: Dict[str, List[float]],
                            performance_values: List[float]) -> SensitivityRanking:
        """Rank by absolute correlation with performance."""
        rankings = []
        
        for param_name, param_values in hyperparameters.items():
            result = self.analyze_correlation(
                param_values, performance_values, param_name, method='spearman'
            )
            # Use absolute correlation as importance score
            rankings.append((param_name, abs(result.correlation)))
        
        # Sort by importance (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return SensitivityRanking(
            rankings=rankings,
            method='correlation'
        )
    
    def _rank_by_variance(self, 
                         hyperparameters: Dict[str, List[float]],
                         performance_values: List[float]) -> SensitivityRanking:
        """Rank by variance explained in performance."""
        rankings = []
        
        perf_arr = np.array(performance_values)
        total_variance = np.var(perf_arr)
        
        for param_name, param_values in hyperparameters.items():
            param_arr = np.array(param_values)
            
            # Compute variance explained using simple linear regression
            # Fit: performance = a * param + b
            coeffs = np.polyfit(param_arr, perf_arr, 1)
            predicted = np.polyval(coeffs, param_arr)
            
            # Variance explained
            explained_variance = 1 - (np.var(perf_arr - predicted) / total_variance)
            explained_variance = max(0.0, explained_variance)  # Clip to [0, 1]
            
            rankings.append((param_name, explained_variance))
        
        # Sort by importance (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return SensitivityRanking(
            rankings=rankings,
            method='variance_explained'
        )
    
    def detect_interactions(self, 
                          param1_values: List[float],
                          param2_values: List[float],
                          performance_values: List[float],
                          param1_name: str,
                          param2_name: str) -> InteractionEffect:
        """
        Detect interaction effects between two hyperparameters.
        
        Args:
            param1_values: Values of first hyperparameter
            param2_values: Values of second hyperparameter
            performance_values: Corresponding performance values
            param1_name: Name of first hyperparameter
            param2_name: Name of second hyperparameter
            
        Returns:
            InteractionEffect object
        """
        p1_arr = np.array(param1_values)
        p2_arr = np.array(param2_values)
        perf_arr = np.array(performance_values)
        
        # Normalize parameters to [0, 1]
        p1_norm = (p1_arr - np.min(p1_arr)) / (np.max(p1_arr) - np.min(p1_arr) + 1e-10)
        p2_norm = (p2_arr - np.min(p2_arr)) / (np.max(p2_arr) - np.min(p2_arr) + 1e-10)
        
        # Create interaction term
        interaction = p1_norm * p2_norm
        
        # Fit model with and without interaction
        # Model 1: perf = a*p1 + b*p2 + c
        X_main = np.column_stack([p1_norm, p2_norm, np.ones(len(p1_norm))])
        coeffs_main = np.linalg.lstsq(X_main, perf_arr, rcond=None)[0]
        pred_main = X_main @ coeffs_main
        
        # Model 2: perf = a*p1 + b*p2 + c*p1*p2 + d
        X_interaction = np.column_stack([p1_norm, p2_norm, interaction, np.ones(len(p1_norm))])
        coeffs_interaction = np.linalg.lstsq(X_interaction, perf_arr, rcond=None)[0]
        pred_interaction = X_interaction @ coeffs_interaction
        
        # Compare models using R² improvement
        ss_total = np.sum((perf_arr - np.mean(perf_arr)) ** 2)
        ss_res_main = np.sum((perf_arr - pred_main) ** 2)
        ss_res_interaction = np.sum((perf_arr - pred_interaction) ** 2)
        
        r2_main = 1 - (ss_res_main / ss_total)
        r2_interaction = 1 - (ss_res_interaction / ss_total)
        
        # Interaction strength = improvement in R²
        interaction_strength = r2_interaction - r2_main
        
        # F-test for significance
        n = len(perf_arr)
        f_stat = ((ss_res_main - ss_res_interaction) / 1) / (ss_res_interaction / (n - 4))
        p_value = 1 - stats.f.cdf(f_stat, 1, n - 4)
        
        is_significant = p_value < self.significance_level
        
        # Interpretation
        if is_significant:
            if interaction_strength > 0:
                interpretation = (f"Significant interaction detected: the effect of {param1_name} "
                                f"depends on the value of {param2_name}")
            else:
                interpretation = f"Weak interaction between {param1_name} and {param2_name}"
        else:
            interpretation = f"No significant interaction between {param1_name} and {param2_name}"
        
        return InteractionEffect(
            param1=param1_name,
            param2=param2_name,
            interaction_strength=float(interaction_strength),
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def identify_optimal_range(self, 
                              hyperparameter_values: List[float],
                              performance_values: List[float],
                              hyperparameter_name: str,
                              percentile: float = 90) -> Dict:
        """
        Identify optimal range for a hyperparameter.
        
        Args:
            hyperparameter_values: Values of the hyperparameter
            performance_values: Corresponding performance values
            hyperparameter_name: Name of the hyperparameter
            percentile: Percentile threshold for "good" performance
            
        Returns:
            Dictionary with optimal range information
        """
        hp_arr = np.array(hyperparameter_values)
        perf_arr = np.array(performance_values)
        
        # Find threshold for good performance
        perf_threshold = np.percentile(perf_arr, percentile)
        
        # Find hyperparameter values that achieve good performance
        good_indices = perf_arr >= perf_threshold
        good_hp_values = hp_arr[good_indices]
        
        if len(good_hp_values) == 0:
            return {
                'hyperparameter': hyperparameter_name,
                'optimal_range': None,
                'optimal_min': None,
                'optimal_max': None,
                'optimal_median': None,
                'n_good_runs': 0,
                'interpretation': f"No runs achieved {percentile}th percentile performance"
            }
        
        optimal_min = float(np.min(good_hp_values))
        optimal_max = float(np.max(good_hp_values))
        optimal_median = float(np.median(good_hp_values))
        
        return {
            'hyperparameter': hyperparameter_name,
            'optimal_range': (optimal_min, optimal_max),
            'optimal_min': optimal_min,
            'optimal_max': optimal_max,
            'optimal_median': optimal_median,
            'n_good_runs': len(good_hp_values),
            'performance_threshold': float(perf_threshold),
            'interpretation': (f"Optimal range for {hyperparameter_name}: "
                             f"[{optimal_min:.4f}, {optimal_max:.4f}] "
                             f"(median: {optimal_median:.4f})")
        }
    
    def analyze_all_hyperparameters(self, 
                                   hyperparameters: Dict[str, List[float]],
                                   performance_values: List[float]) -> Dict:
        """
        Comprehensive analysis of all hyperparameters.
        
        Args:
            hyperparameters: Dictionary mapping hyperparameter names to values
            performance_values: Corresponding performance values
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        results = {
            'correlations': {},
            'sensitivity_ranking': None,
            'optimal_ranges': {},
            'summary': {}
        }
        
        # Correlation analysis for each hyperparameter
        for param_name, param_values in hyperparameters.items():
            corr_result = self.analyze_correlation(
                param_values, performance_values, param_name
            )
            results['correlations'][param_name] = corr_result
        
        # Sensitivity ranking
        results['sensitivity_ranking'] = self.rank_sensitivity(
            hyperparameters, performance_values
        )
        
        # Optimal ranges
        for param_name, param_values in hyperparameters.items():
            optimal_range = self.identify_optimal_range(
                param_values, performance_values, param_name
            )
            results['optimal_ranges'][param_name] = optimal_range
        
        # Summary
        most_important = results['sensitivity_ranking'].rankings[0] if results['sensitivity_ranking'].rankings else None
        significant_params = [name for name, corr in results['correlations'].items() if corr.is_significant]
        
        results['summary'] = {
            'most_important_hyperparameter': most_important[0] if most_important else None,
            'n_significant_correlations': len(significant_params),
            'significant_hyperparameters': significant_params
        }
        
        return results

