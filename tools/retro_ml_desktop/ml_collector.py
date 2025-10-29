#!/usr/bin/env python3
"""
ML Metrics Collection System

Real-time collection and parsing of training metrics from various sources:
- Training process logs
- TensorBoard logs
- System resource monitoring
- Custom metric extraction
"""

import re
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging
import psutil
import subprocess

from .ml_metrics import TrainingMetrics, ExperimentRun, ExperimentConfig
from .ml_database import MetricsDatabase


class LogParser:
    """
    Advanced log parser for extracting ML training metrics.
    
    Supports multiple log formats:
    - Stable-Baselines3 training logs
    - TensorBoard scalar logs
    - Custom training callbacks
    - System resource logs
    """
    
    def __init__(self):
        """Initialize log parser with regex patterns."""
        # Training progress patterns
        self.progress_patterns = {
            # [Stats] Progress: 10,000/4,000,000 (0.3%) | Speed: 1000 steps/s | ETA: 1.1h
            'sb3_progress': re.compile(
                r'\[Stats\] Progress:\s*([\d,]+)/([\d,]+)\s*\(([\d.]+)%\)\s*\|\s*Speed:\s*([\d.]+)\s*steps/s\s*\|\s*ETA:\s*([\d.]+)h'
            ),
            
            # [Milestone] Milestone reached: 10% at step 400,000
            'milestone': re.compile(
                r'\[Milestone\] Milestone reached:\s*([\d.]+)%\s*at step\s*([\d,]+)'
            ),
            
            # [Start] Training started - Target: 4,000,000 timesteps
            'training_start': re.compile(
                r'\[Start\] Training started - Target:\s*([\d,]+)\s*timesteps'
            )
        }
        
        # Loss and reward patterns
        self.metric_patterns = {
            # ep_rew_mean | 5.64
            'episode_reward': re.compile(
                r'ep_rew_mean\s*[|\s]*([+-]?\d*\.?\d+)'
            ),
            
            # policy_loss | 0.0234
            'policy_loss': re.compile(
                r'policy_loss\s*[|\s]*([+-]?\d*\.?\d+)'
            ),
            
            # value_loss | 0.0156
            'value_loss': re.compile(
                r'value_loss\s*[|\s]*([+-]?\d*\.?\d+)'
            ),
            
            # entropy_loss | 0.0089
            'entropy_loss': re.compile(
                r'entropy_loss\s*[|\s]*([+-]?\d*\.?\d+)'
            ),
            
            # learning_rate | 0.0003
            'learning_rate': re.compile(
                r'learning_rate\s*[|\s]*([+-]?\d*\.?\d+)'
            ),
            
            # kl_divergence | 0.0012
            'kl_divergence': re.compile(
                r'kl_divergence\s*[|\s]*([+-]?\d*\.?\d+)'
            ),
            
            # explained_variance | 0.85
            'explained_variance': re.compile(
                r'explained_variance\s*[|\s]*([+-]?\d*\.?\d+)'
            )
        }
    
    def parse_log_chunk(self, log_text: str, run_id: str) -> List[TrainingMetrics]:
        """
        Parse a chunk of log text and extract training metrics.
        
        Args:
            log_text: Raw log text to parse
            run_id: Run ID for the metrics
            
        Returns:
            List of TrainingMetrics objects extracted from logs
        """
        metrics_list = []
        lines = log_text.split('\n')
        
        # Current metric being built
        current_metric = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for progress updates
            for pattern_name, pattern in self.progress_patterns.items():
                match = pattern.search(line)
                if match:
                    if pattern_name == 'sb3_progress':
                        current_steps = int(match.group(1).replace(',', ''))
                        total_steps = int(match.group(2).replace(',', ''))
                        progress_pct = float(match.group(3))
                        fps = float(match.group(4))
                        eta_hours = float(match.group(5))
                        
                        # Create new metric if we have progress data
                        current_metric = TrainingMetrics(
                            run_id=run_id,
                            timestep=current_steps,
                            timestamp=datetime.now(),
                            episode_count=0,  # Will be updated if found
                            total_timesteps=total_steps,
                            progress_pct=progress_pct,
                            fps=fps,
                            steps_per_second=fps
                        )
                        
                    elif pattern_name == 'milestone':
                        milestone_pct = float(match.group(1))
                        milestone_step = int(match.group(2).replace(',', ''))
                        
                        # Update current metric or create new one
                        if current_metric:
                            current_metric.timestep = milestone_step
                            current_metric.progress_pct = milestone_pct
                        else:
                            current_metric = TrainingMetrics(
                                run_id=run_id,
                                timestep=milestone_step,
                                timestamp=datetime.now(),
                                episode_count=0,
                                total_timesteps=0,  # Will be updated
                                progress_pct=milestone_pct
                            )
                    
                    elif pattern_name == 'training_start':
                        total_steps = int(match.group(1).replace(',', ''))
                        if current_metric:
                            current_metric.total_timesteps = total_steps
            
            # Check for metric values
            for metric_name, pattern in self.metric_patterns.items():
                match = pattern.search(line)
                if match:
                    value = float(match.group(1))
                    
                    # Update current metric
                    if current_metric:
                        if metric_name == 'episode_reward':
                            current_metric.episode_reward_mean = value
                        elif metric_name == 'policy_loss':
                            current_metric.policy_loss = value
                        elif metric_name == 'value_loss':
                            current_metric.value_loss = value
                        elif metric_name == 'entropy_loss':
                            current_metric.entropy_loss = value
                        elif metric_name == 'learning_rate':
                            current_metric.learning_rate = value
                        elif metric_name == 'kl_divergence':
                            current_metric.kl_divergence = value
                        elif metric_name == 'explained_variance':
                            current_metric.explained_variance = value
        
        # Add completed metric to list
        if current_metric and current_metric.timestep > 0:
            metrics_list.append(current_metric)
        
        return metrics_list


class SystemMonitor:
    """
    System resource monitoring for training processes.
    
    Tracks CPU, memory, and GPU utilization during training.
    """
    
    def __init__(self):
        """Initialize system monitor."""
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def get_system_metrics(self, pid: Optional[int] = None) -> Dict[str, float]:
        """
        Get current system resource metrics.
        
        Args:
            pid: Process ID to monitor (if None, monitors system-wide)
            
        Returns:
            Dictionary with resource metrics
        """
        metrics = {}
        
        try:
            if pid:
                # Process-specific metrics
                process = psutil.Process(pid)
                metrics['cpu_percent'] = process.cpu_percent()
                metrics['memory_mb'] = process.memory_info().rss / 1024 / 1024
            else:
                # System-wide metrics
                metrics['cpu_percent'] = psutil.cpu_percent()
                metrics['memory_mb'] = psutil.virtual_memory().used / 1024 / 1024
            
            # GPU metrics (if available)
            if self.gpu_available:
                gpu_metrics = self._get_gpu_metrics()
                metrics.update(gpu_metrics)
        
        except Exception as e:
            logging.warning(f"Failed to get system metrics: {e}")
        
        return metrics
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU utilization metrics."""
        try:
            # GPU utilization
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_percent = float(result.stdout.strip())
            else:
                gpu_percent = 0.0
            
            # GPU memory
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_memory_mb = float(result.stdout.strip())
            else:
                gpu_memory_mb = 0.0
            
            return {
                'gpu_percent': gpu_percent,
                'gpu_memory_mb': gpu_memory_mb
            }
        
        except Exception as e:
            logging.warning(f"Failed to get GPU metrics: {e}")
            return {'gpu_percent': 0.0, 'gpu_memory_mb': 0.0}


class MetricsCollector:
    """
    Real-time metrics collection system for ML experiments.
    
    Coordinates log parsing, system monitoring, and database storage
    for comprehensive experiment tracking.
    """
    
    def __init__(self, database: MetricsDatabase):
        """
        Initialize metrics collector.
        
        Args:
            database: MetricsDatabase instance for storage
        """
        self.database = database
        self.log_parser = LogParser()
        self.system_monitor = SystemMonitor()
        
        # Active collectors
        self.active_collectors = {}  # run_id -> collector thread
        self.stop_events = {}  # run_id -> stop event
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def start_collection(self, run_id: str, log_source: Callable[[], str], 
                        pid: Optional[int] = None, interval: float = 10.0):
        """
        Start real-time metrics collection for a training run.
        
        Args:
            run_id: Unique run identifier
            log_source: Function that returns current log content
            pid: Process ID for system monitoring
            interval: Collection interval in seconds
        """
        if run_id in self.active_collectors:
            self.logger.warning(f"Collection already active for run {run_id}")
            return
        
        # Create stop event
        stop_event = threading.Event()
        self.stop_events[run_id] = stop_event
        
        # Start collector thread
        collector_thread = threading.Thread(
            target=self._collection_loop,
            args=(run_id, log_source, pid, interval, stop_event),
            daemon=True
        )
        
        self.active_collectors[run_id] = collector_thread
        collector_thread.start()
        
        self.logger.info(f"Started metrics collection for run {run_id}")
    
    def stop_collection(self, run_id: str):
        """
        Stop metrics collection for a training run.
        
        Args:
            run_id: Run identifier to stop collection for
        """
        if run_id not in self.active_collectors:
            return
        
        # Signal stop
        self.stop_events[run_id].set()
        
        # Wait for thread to finish
        self.active_collectors[run_id].join(timeout=5.0)
        
        # Cleanup
        del self.active_collectors[run_id]
        del self.stop_events[run_id]
        
        self.logger.info(f"Stopped metrics collection for run {run_id}")
    
    def _collection_loop(self, run_id: str, log_source: Callable[[], str], 
                        pid: Optional[int], interval: float, stop_event: threading.Event):
        """
        Main collection loop for a training run.
        
        Args:
            run_id: Run identifier
            log_source: Function to get log content
            pid: Process ID for monitoring
            interval: Collection interval
            stop_event: Event to signal stop
        """
        last_log_position = 0
        
        while not stop_event.wait(interval):
            try:
                # Get new log content
                current_logs = log_source()
                
                # Only process new log content
                if len(current_logs) > last_log_position:
                    new_logs = current_logs[last_log_position:]
                    last_log_position = len(current_logs)
                    
                    # Parse metrics from new logs
                    metrics_list = self.log_parser.parse_log_chunk(new_logs, run_id)
                    
                    # Add system metrics to latest metric
                    if metrics_list:
                        system_metrics = self.system_monitor.get_system_metrics(pid)
                        latest_metric = metrics_list[-1]
                        
                        # Update with system metrics
                        for key, value in system_metrics.items():
                            setattr(latest_metric, key, value)
                        
                        # Store metrics in database
                        for metric in metrics_list:
                            self.database.add_training_metrics(metric)

                        # Update experiment run with latest progress and best reward
                        if metrics_list:
                            latest_metric = metrics_list[-1]
                            updates = {}

                            if latest_metric.timestep:
                                updates['current_timestep'] = latest_metric.timestep

                            # Note: experiment_runs table doesn't have progress_pct column
                            # Progress can be calculated from current_timestep / total_timesteps

                            if latest_metric.episode_reward_mean is not None:
                                # Get current best reward to compare
                                current_best = self.database.get_experiment_run_field(run_id, 'best_reward')
                                if current_best is None or latest_metric.episode_reward_mean > current_best:
                                    updates['best_reward'] = latest_metric.episode_reward_mean

                            if updates:
                                self.database.update_experiment_run(run_id, **updates)

                        self.logger.debug(f"Collected {len(metrics_list)} metrics for run {run_id}")
            
            except Exception as e:
                self.logger.error(f"Error in collection loop for {run_id}: {e}")
    
    def get_active_runs(self) -> List[str]:
        """Get list of runs with active collection."""
        return list(self.active_collectors.keys())
    
    def stop_all_collection(self):
        """Stop all active metric collection."""
        for run_id in list(self.active_collectors.keys()):
            self.stop_collection(run_id)
