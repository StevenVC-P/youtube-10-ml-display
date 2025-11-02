#!/usr/bin/env python3
"""
Statistical Analysis Module for ML Training Runs

Provides comprehensive statistical analysis of multiple training runs:
- Descriptive statistics (mean, median, std, quartiles)
- Confidence intervals
- Hypothesis testing
- Distribution analysis
- Outlier detection

Part of Sprint 2: Advanced Metrics & Analytics
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class StatisticalSummary:
    """Statistical summary of multiple runs."""
    
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    iqr: float  # Interquartile range
    cv: float   # Coefficient of variation
    n_samples: int
    
    def __str__(self):
        return (f"Mean: {self.mean:.4f} ± {self.std:.4f}\n"
               f"Median: {self.median:.4f} (IQR: {self.iqr:.4f})\n"
               f"Range: [{self.min:.4f}, {self.max:.4f}]\n"
               f"CV: {self.cv:.2%}, N: {self.n_samples}")


@dataclass
class ConfidenceInterval:
    """Confidence interval for a statistic."""
    
    lower: float
    upper: float
    confidence_level: float
    method: str
    
    def __str__(self):
        return f"[{self.lower:.4f}, {self.upper:.4f}] ({self.confidence_level:.0%} CI, {self.method})"


@dataclass
class ComparisonResult:
    """Result of statistical comparison between runs."""
    
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    interpretation: str
    
    def __str__(self):
        sig = "significant" if self.is_significant else "not significant"
        return (f"{self.test_name}: statistic={self.statistic:.4f}, "
               f"p={self.p_value:.4f} ({sig} at α={self.alpha})\n"
               f"{self.interpretation}")


class StatisticalAnalyzer:
    """
    Performs statistical analysis on ML training runs.
    
    Provides methods for:
    - Descriptive statistics
    - Confidence intervals
    - Hypothesis testing
    - Distribution analysis
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
    
    def compute_summary(self, values: List[float]) -> StatisticalSummary:
        """
        Compute comprehensive statistical summary.
        
        Args:
            values: List of values to analyze
            
        Returns:
            StatisticalSummary object
        """
        arr = np.array(values)
        
        mean = np.mean(arr)
        median = np.median(arr)
        std = np.std(arr, ddof=1)  # Sample std
        min_val = np.min(arr)
        max_val = np.max(arr)
        q25 = np.percentile(arr, 25)
        q75 = np.percentile(arr, 75)
        iqr = q75 - q25
        cv = std / mean if mean != 0 else 0.0
        
        return StatisticalSummary(
            mean=float(mean),
            median=float(median),
            std=float(std),
            min=float(min_val),
            max=float(max_val),
            q25=float(q25),
            q75=float(q75),
            iqr=float(iqr),
            cv=float(cv),
            n_samples=len(arr)
        )
    
    def compute_confidence_interval(self, 
                                   values: List[float],
                                   method: str = 'bootstrap') -> ConfidenceInterval:
        """
        Compute confidence interval for the mean.
        
        Args:
            values: List of values
            method: Method to use ('t-test', 'bootstrap')
            
        Returns:
            ConfidenceInterval object
        """
        arr = np.array(values)
        
        if method == 't-test':
            return self._ci_t_test(arr)
        elif method == 'bootstrap':
            return self._ci_bootstrap(arr)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _ci_t_test(self, arr: np.ndarray) -> ConfidenceInterval:
        """Compute confidence interval using t-distribution."""
        mean = np.mean(arr)
        sem = stats.sem(arr)  # Standard error of mean
        
        # Degrees of freedom
        df = len(arr) - 1
        
        # Critical value
        alpha = 1 - self.confidence_level
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        # Confidence interval
        margin = t_crit * sem
        
        return ConfidenceInterval(
            lower=float(mean - margin),
            upper=float(mean + margin),
            confidence_level=self.confidence_level,
            method='t-test'
        )
    
    def _ci_bootstrap(self, 
                     arr: np.ndarray, 
                     n_bootstrap: int = 10000) -> ConfidenceInterval:
        """Compute confidence interval using bootstrap."""
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Percentile method
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return ConfidenceInterval(
            lower=float(lower),
            upper=float(upper),
            confidence_level=self.confidence_level,
            method='bootstrap'
        )
    
    def compare_runs(self, 
                    run1_values: List[float],
                    run2_values: List[float],
                    test: str = 'auto',
                    alpha: float = 0.05) -> ComparisonResult:
        """
        Compare two runs statistically.
        
        Args:
            run1_values: Values from first run
            run2_values: Values from second run
            test: Test to use ('auto', 't-test', 'mann-whitney', 'ks')
            alpha: Significance level
            
        Returns:
            ComparisonResult object
        """
        arr1 = np.array(run1_values)
        arr2 = np.array(run2_values)
        
        if test == 'auto':
            # Check normality
            _, p1 = stats.shapiro(arr1) if len(arr1) < 5000 else (0, 1)
            _, p2 = stats.shapiro(arr2) if len(arr2) < 5000 else (0, 1)
            
            # Use t-test if both normal, otherwise Mann-Whitney
            if p1 > 0.05 and p2 > 0.05:
                test = 't-test'
            else:
                test = 'mann-whitney'
        
        if test == 't-test':
            return self._compare_t_test(arr1, arr2, alpha)
        elif test == 'mann-whitney':
            return self._compare_mann_whitney(arr1, arr2, alpha)
        elif test == 'ks':
            return self._compare_ks(arr1, arr2, alpha)
        else:
            raise ValueError(f"Unknown test: {test}")
    
    def _compare_t_test(self, 
                       arr1: np.ndarray, 
                       arr2: np.ndarray, 
                       alpha: float) -> ComparisonResult:
        """Compare using independent t-test."""
        statistic, p_value = stats.ttest_ind(arr1, arr2)
        
        is_significant = p_value < alpha
        
        mean1 = np.mean(arr1)
        mean2 = np.mean(arr2)
        
        if is_significant:
            if mean1 > mean2:
                interpretation = f"Run 1 (mean={mean1:.4f}) is significantly higher than Run 2 (mean={mean2:.4f})"
            else:
                interpretation = f"Run 2 (mean={mean2:.4f}) is significantly higher than Run 1 (mean={mean1:.4f})"
        else:
            interpretation = f"No significant difference between runs (mean1={mean1:.4f}, mean2={mean2:.4f})"
        
        return ComparisonResult(
            test_name='Independent t-test',
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
    
    def _compare_mann_whitney(self, 
                             arr1: np.ndarray, 
                             arr2: np.ndarray, 
                             alpha: float) -> ComparisonResult:
        """Compare using Mann-Whitney U test."""
        statistic, p_value = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')
        
        is_significant = p_value < alpha
        
        median1 = np.median(arr1)
        median2 = np.median(arr2)
        
        if is_significant:
            if median1 > median2:
                interpretation = f"Run 1 (median={median1:.4f}) is significantly higher than Run 2 (median={median2:.4f})"
            else:
                interpretation = f"Run 2 (median={median2:.4f}) is significantly higher than Run 1 (median={median1:.4f})"
        else:
            interpretation = f"No significant difference between runs (median1={median1:.4f}, median2={median2:.4f})"
        
        return ComparisonResult(
            test_name='Mann-Whitney U test',
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
    
    def _compare_ks(self, 
                   arr1: np.ndarray, 
                   arr2: np.ndarray, 
                   alpha: float) -> ComparisonResult:
        """Compare using Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(arr1, arr2)
        
        is_significant = p_value < alpha
        
        if is_significant:
            interpretation = "The distributions of the two runs are significantly different"
        else:
            interpretation = "The distributions of the two runs are not significantly different"
        
        return ComparisonResult(
            test_name='Kolmogorov-Smirnov test',
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
    
    def detect_outliers(self, 
                       values: List[float],
                       method: str = 'iqr') -> Tuple[List[int], List[float]]:
        """
        Detect outliers in data.
        
        Args:
            values: List of values
            method: Detection method ('iqr', 'zscore')
            
        Returns:
            Tuple of (outlier_indices, outlier_values)
        """
        arr = np.array(values)
        
        if method == 'iqr':
            q25 = np.percentile(arr, 25)
            q75 = np.percentile(arr, 75)
            iqr = q75 - q25
            
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outlier_mask = (arr < lower_bound) | (arr > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(arr))
            outlier_mask = z_scores > 3
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = arr[outlier_mask].tolist()
        
        return outlier_indices, outlier_values
    
    def analyze_distribution(self, values: List[float]) -> Dict:
        """
        Analyze the distribution of values.
        
        Args:
            values: List of values
            
        Returns:
            Dictionary with distribution analysis
        """
        arr = np.array(values)
        
        # Normality test
        if len(arr) < 5000:
            shapiro_stat, shapiro_p = stats.shapiro(arr)
        else:
            shapiro_stat, shapiro_p = None, None
        
        # Skewness and kurtosis
        skewness = stats.skew(arr)
        kurtosis = stats.kurtosis(arr)
        
        return {
            'is_normal': shapiro_p > 0.05 if shapiro_p is not None else None,
            'shapiro_statistic': float(shapiro_stat) if shapiro_stat is not None else None,
            'shapiro_p_value': float(shapiro_p) if shapiro_p is not None else None,
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'interpretation': self._interpret_distribution(skewness, kurtosis, shapiro_p)
        }
    
    def _interpret_distribution(self, 
                               skewness: float, 
                               kurtosis: float, 
                               shapiro_p: Optional[float]) -> str:
        """Interpret distribution characteristics."""
        parts = []
        
        if shapiro_p is not None:
            if shapiro_p > 0.05:
                parts.append("Distribution appears normal")
            else:
                parts.append("Distribution is non-normal")
        
        if abs(skewness) < 0.5:
            parts.append("approximately symmetric")
        elif skewness > 0:
            parts.append("right-skewed (positive skew)")
        else:
            parts.append("left-skewed (negative skew)")
        
        if abs(kurtosis) < 0.5:
            parts.append("mesokurtic (normal tail)")
        elif kurtosis > 0:
            parts.append("leptokurtic (heavy tails)")
        else:
            parts.append("platykurtic (light tails)")
        
        return ", ".join(parts)

