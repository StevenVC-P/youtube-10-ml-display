#!/usr/bin/env python3
"""
ML Metrics and Experiment Tracking System

This module provides comprehensive data models and tracking capabilities
for machine learning experiments, designed for ML scientists who need
detailed analysis and comparison of training runs.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np


@dataclass
class TrainingMetrics:
    """
    Core training metrics captured at each timestep.
    
    Designed to capture all essential ML training information
    for comprehensive analysis and visualization.
    """
    # Core identifiers
    run_id: str
    timestep: int
    timestamp: datetime
    
    # Training progress
    episode_count: int
    total_timesteps: int
    progress_pct: float
    
    # Performance metrics
    episode_reward_mean: Optional[float] = None
    episode_reward_std: Optional[float] = None
    episode_length_mean: Optional[float] = None
    
    # Algorithm-specific losses
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy_loss: Optional[float] = None
    total_loss: Optional[float] = None
    
    # Learning dynamics
    learning_rate: Optional[float] = None
    kl_divergence: Optional[float] = None
    clip_fraction: Optional[float] = None
    explained_variance: Optional[float] = None
    
    # System performance
    fps: Optional[float] = None
    steps_per_second: Optional[float] = None
    
    # Resource utilization
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    
    # Training stability
    gradient_norm: Optional[float] = None
    weight_norm: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO string
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        """Create from dictionary (e.g., from JSON)."""
        # Convert ISO string back to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration for reproducibility.

    Captures all hyperparameters, environment settings, and
    model architecture details needed to reproduce experiments.
    """
    # Required fields first
    algorithm: str
    policy_type: str
    env_id: str
    n_envs: int
    learning_rate: float
    batch_size: int
    n_steps: int
    gamma: float
    frame_stack: int
    action_repeat: int
    total_timesteps: int
    eval_freq: int
    save_freq: int

    # Optional fields with defaults
    gae_lambda: Optional[float] = None
    clip_range: Optional[float] = None
    ent_coef: Optional[float] = None
    vf_coef: Optional[float] = None
    max_grad_norm: Optional[float] = None
    max_episode_steps: Optional[int] = None
    
    # Model architecture
    net_arch: Optional[List[int]] = None
    activation_fn: str = "tanh"
    
    # System configuration
    device: str = "auto"
    seed: Optional[int] = None
    
    # Custom parameters
    custom_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentRun:
    """
    Complete experiment run with metadata, configuration, and results.
    
    Represents a single training run with all associated data
    for comprehensive analysis and comparison.
    """
    # Core identifiers (required fields first)
    run_id: str
    experiment_name: str
    start_time: datetime

    # Optional identifiers
    custom_name: Optional[str] = None
    leg_number: int = 1
    base_run_id: Optional[str] = None

    # Timing
    end_time: Optional[datetime] = None

    # Status tracking
    status: str = "running"  # running, paused, stopped, completed, failed
    current_timestep: int = 0
    leg_start_timestep: int = 0  # Timestep where this leg started
    
    # Configuration
    config: ExperimentConfig = None
    
    # Results summary
    best_reward: Optional[float] = None
    final_reward: Optional[float] = None
    convergence_timestep: Optional[int] = None
    
    # File paths
    model_path: Optional[str] = None
    log_path: Optional[str] = None
    video_path: Optional[str] = None
    tensorboard_path: Optional[str] = None
    
    # Metadata
    git_commit: Optional[str] = None
    python_version: Optional[str] = None
    dependencies: Optional[Dict[str, str]] = None
    
    # Notes and tags
    description: Optional[str] = None
    status_note: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetimes to ISO strings
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRun':
        """Create from dictionary."""
        # Convert ISO strings back to datetimes
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        # Convert config dict back to ExperimentConfig
        if data.get('config') and isinstance(data['config'], dict):
            data['config'] = ExperimentConfig.from_dict(data['config'])
        
        return cls(**data)
    
    @property
    def duration(self) -> Optional[float]:
        """Get training duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        else:
            return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        return self.status in ["running", "paused"]
    
    @property
    def is_completed(self) -> bool:
        """Check if experiment has completed successfully."""
        return self.status == "completed"
    
    def update_status(self, status: str, timestep: Optional[int] = None):
        """Update experiment status and current timestep."""
        self.status = status
        if timestep is not None:
            self.current_timestep = timestep
        
        if status in ["completed", "failed", "stopped"]:
            self.end_time = datetime.now()


class MetricsAggregator:
    """
    Utility class for aggregating and analyzing training metrics.
    
    Provides statistical analysis and comparison capabilities
    for ML experiment evaluation.
    """
    
    @staticmethod
    def calculate_moving_average(values: List[float], window: int = 100) -> List[float]:
        """Calculate moving average with specified window size."""
        if len(values) < window:
            return values
        
        result = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i + 1]
            result.append(np.mean(window_values))
        
        return result
    
    @staticmethod
    def detect_convergence(rewards: List[float], window: int = 1000, threshold: float = 0.01) -> Optional[int]:
        """
        Detect convergence point in training rewards.
        
        Returns the timestep where training appears to have converged,
        or None if convergence is not detected.
        """
        if len(rewards) < window * 2:
            return None
        
        # Calculate moving averages
        moving_avg = MetricsAggregator.calculate_moving_average(rewards, window)
        
        # Look for stability (low variance) in recent window
        for i in range(window, len(moving_avg) - window):
            recent_window = moving_avg[i:i + window]
            variance = np.var(recent_window)
            
            if variance < threshold:
                return i
        
        return None
    
    @staticmethod
    def compare_runs(run1_metrics: List[TrainingMetrics], 
                    run2_metrics: List[TrainingMetrics]) -> Dict[str, Any]:
        """
        Compare two training runs and return statistical analysis.
        
        Returns comprehensive comparison including convergence,
        sample efficiency, and final performance.
        """
        # Extract rewards
        rewards1 = [m.episode_reward_mean for m in run1_metrics if m.episode_reward_mean is not None]
        rewards2 = [m.episode_reward_mean for m in run2_metrics if m.episode_reward_mean is not None]
        
        if not rewards1 or not rewards2:
            return {"error": "Insufficient reward data for comparison"}
        
        # Basic statistics
        comparison = {
            "run1_final_reward": rewards1[-1] if rewards1 else None,
            "run2_final_reward": rewards2[-1] if rewards2 else None,
            "run1_best_reward": max(rewards1) if rewards1 else None,
            "run2_best_reward": max(rewards2) if rewards2 else None,
            "run1_mean_reward": np.mean(rewards1),
            "run2_mean_reward": np.mean(rewards2),
            "run1_std_reward": np.std(rewards1),
            "run2_std_reward": np.std(rewards2),
        }
        
        # Convergence analysis
        conv1 = MetricsAggregator.detect_convergence(rewards1)
        conv2 = MetricsAggregator.detect_convergence(rewards2)
        
        comparison.update({
            "run1_convergence_step": conv1,
            "run2_convergence_step": conv2,
            "faster_convergence": "run1" if conv1 and conv2 and conv1 < conv2 else "run2" if conv1 and conv2 else None
        })
        
        # Sample efficiency (reward at different timesteps)
        milestones = [0.1, 0.25, 0.5, 0.75, 1.0]
        for milestone in milestones:
            idx1 = int(milestone * len(rewards1)) - 1 if len(rewards1) > 0 else 0
            idx2 = int(milestone * len(rewards2)) - 1 if len(rewards2) > 0 else 0
            
            if idx1 >= 0 and idx1 < len(rewards1):
                comparison[f"run1_reward_at_{int(milestone*100)}pct"] = rewards1[idx1]
            if idx2 >= 0 and idx2 < len(rewards2):
                comparison[f"run2_reward_at_{int(milestone*100)}pct"] = rewards2[idx2]
        
        return comparison
