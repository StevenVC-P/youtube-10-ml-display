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
import json
import yaml
import os

from .ml_metrics import TrainingMetrics, ExperimentRun, ExperimentConfig
from .ml_database import MetricsDatabase

DEBUG_METRICS_INGESTION = os.getenv("DEBUG_METRICS_INGESTION") in ("1", "true", "True", "yes")


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
        
        # Loss and reward patterns - Updated for Stable-Baselines3 table format
        self.metric_patterns = {
            # |    ep_rew_mean         | 5.64         |
            'episode_reward': re.compile(
                r'\|\s*ep_rew_mean\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            ),

            # |    policy_gradient_loss | -0.00158     |
            'policy_loss': re.compile(
                r'\|\s*policy_gradient_loss\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            ),

            # |    value_loss           | 0.295        |
            'value_loss': re.compile(
                r'\|\s*value_loss\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            ),

            # |    entropy_loss         | 0.0089       |
            'entropy_loss': re.compile(
                r'\|\s*entropy_loss\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            ),

            # |    learning_rate        | 0.0003       |
            'learning_rate': re.compile(
                r'\|\s*learning_rate\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            ),

            # |    kl_divergence        | 0.0012       |
            'kl_divergence': re.compile(
                r'\|\s*kl_divergence\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            ),

            # |    explained_variance   | 0.85         |
            'explained_variance': re.compile(
                r'\|\s*explained_variance\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            ),

            # Additional SB3 metrics
            # |    fps                  | 631          |
            'fps': re.compile(
                r'\|\s*fps\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            ),

            # |    total_timesteps      | 1024         |
            'total_timesteps': re.compile(
                r'\|\s*total_timesteps\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            ),

            # |    iterations           | 1            |
            'iterations': re.compile(
                r'\|\s*iterations\s*\|\s*([+-]?\d*\.?\d+)\s*\|'
            )
        }

        # Warn only once per run when target timesteps are unknown
        self._warned_missing_target = set()
    
    def parse_log_chunk(self, log_text: str, run_id: str, target_total_timesteps: Optional[int] = None) -> List[TrainingMetrics]:
        """
        Parse a chunk of log text and extract training metrics.
        
        Args:
            log_text: Raw log text to parse
            run_id: Run ID for the metrics
            target_total_timesteps: Expected total timesteps for this run (if known)
            
        Returns:
            List of TrainingMetrics objects extracted from logs
        """
        metrics_list = []
        lines = log_text.split('\n')
        logging.debug(f"Parsing {len(lines)} lines for {run_id}")

        # Current metric being built
        current_metric = None
        saw_stats_line = False

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for progress updates
            for pattern_name, pattern in self.progress_patterns.items():
                match = pattern.search(line)
                if match:
                    if '[Stats]' in line:
                        saw_stats_line = True
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
                    logging.debug(f"Found {metric_name}={value} in line: {line[:100]}")
                elif 'fps' in line and metric_name == 'fps':
                    logging.debug(f"FPS pattern failed to match line: {line}")
                elif 'total_timesteps' in line and metric_name == 'total_timesteps':
                    logging.debug(f"Total timesteps pattern failed to match line: {line}")
                elif 'value_loss' in line and metric_name == 'value_loss':
                    logging.debug(f"Value loss pattern failed to match line: {line}")

                if match:
                    value = float(match.group(1))

                    # Create new metric if we don't have one yet
                    if not current_metric:
                        current_metric = TrainingMetrics(
                            run_id=run_id,
                            timestep=0,  # Will be updated when we find timesteps
                            timestamp=datetime.now(),
                            episode_count=0,
                            total_timesteps=0,
                            progress_pct=0.0
                        )

                    # Update current metric
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
                    elif metric_name == 'fps':
                        current_metric.fps = value
                        current_metric.steps_per_second = value
                    elif metric_name == 'total_timesteps':
                        # This is the current timestep, not the target total
                        current_metric.timestep = int(value)
                    elif metric_name == 'iterations':
                        current_metric.episode_count = int(value)
        
        # Calculate progress percentage if we have timestep info
        if current_metric and current_metric.timestep > 0:
            # Prefer target_total_timesteps passed from caller, otherwise use what we saw in logs
            target_total = target_total_timesteps or current_metric.total_timesteps

            if target_total and target_total > 0:
                current_metric.total_timesteps = target_total
                current_metric.progress_pct = (current_metric.timestep / target_total) * 100.0
            else:
                # Store steps but avoid misleading percentage/ETA
                current_metric.total_timesteps = current_metric.timestep
                current_metric.progress_pct = 0.0

                if run_id not in self._warned_missing_target:
                    logging.warning(
                        f"[Metrics] Missing target timesteps for {run_id}; "
                        "storing step counts only and skipping progress percentage."
                    )
                    self._warned_missing_target.add(run_id)

            metrics_list.append(current_metric)

            # Debug logging
            logging.debug(f"Parsed metric for {run_id}: timestep={current_metric.timestep}, "
                        f"progress={current_metric.progress_pct:.2f}%, "
                        f"reward={current_metric.episode_reward_mean}, "
                        f"fps={current_metric.fps}")

        if DEBUG_METRICS_INGESTION and saw_stats_line and not metrics_list:
            stats_lines = [l for l in lines if '[Stats]' in l]
            logging.warning(f"[METRICS DEBUG] Saw [Stats] but parsed 0 metrics for {run_id}. Sample lines: {stats_lines[:3]}")

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
        self._debug_last_log_time: Dict[str, float] = {}
        
        # Cache target timesteps per run to avoid repeated DB lookups
        self._target_timesteps_cache: Dict[str, Optional[int]] = {}
        
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

    def _get_target_total_timesteps(self, run_id: str) -> Optional[int]:
        """
        Get the expected total timesteps for a run from the database/config.

        We avoid hard-coded defaults and instead use stored run config or
        previously parsed metrics.
        """
        if run_id in self._target_timesteps_cache:
            return self._target_timesteps_cache[run_id]

        target_total = None

        # Prefer total_timesteps saved with the run config
        try:
            config_json = self.database.get_experiment_run_field(run_id, 'config_json')
            if config_json:
                config_data = json.loads(config_json)
                target_total = (
                    config_data.get('total_timesteps')
                    or config_data.get('train', {}).get('total_timesteps')
                    or config_data.get('training', {}).get('total_timesteps')
                )
            if not target_total:
                # Check explicit target_timestep column if present
                target_from_db = self.database.get_experiment_run_field(run_id, 'target_timestep')
                if target_from_db:
                    target_total = int(target_from_db)
        except Exception as e:
            self.logger.warning(f"Could not read target timesteps for {run_id} from config: {e}")

        # Next: read persisted run_metadata.json for absolute targets
        if not target_total:
            try:
                meta_path = Path("models") / "checkpoints" / run_id / "run_metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    target_total = meta.get('target_timestep') or meta.get('total_timesteps')
            except Exception as e:
                self.logger.debug(f"Could not read target timesteps for {run_id} from run_metadata.json: {e}")

        # Fallback: read persisted effective config written alongside checkpoints
        if not target_total:
            try:
                cfg_path = Path("models") / "checkpoints" / run_id / "config_effective.yaml"
                if cfg_path.exists():
                    cfg_data = yaml.safe_load(cfg_path.read_text())
                    target_total = (
                        cfg_data.get('train', {}).get('target_total_timesteps')
                        or cfg_data.get('train', {}).get('total_timesteps')
                    )
            except Exception as e:
                self.logger.debug(f"Could not read target timesteps for {run_id} from config_effective.yaml: {e}")

        # Fallback: use latest stored metrics if they already contain a total
        if not target_total:
            latest_metric = self.database.get_latest_metrics(run_id)
            if latest_metric and latest_metric.total_timesteps:
                target_total = latest_metric.total_timesteps

        self._target_timesteps_cache[run_id] = target_total
        return target_total
    
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
        last_log_hash = None

        while not stop_event.wait(interval):
            try:
                # Get recent log content (log_source returns last 2000 chars)
                current_logs = log_source()
                # Reduced logging - only log at debug level
                self.logger.debug(f"Collection loop for {run_id}: got {len(current_logs)} chars from log_source")

                # Check if logs have changed by comparing hash
                import hashlib
                current_hash = hashlib.md5(current_logs.encode()).hexdigest() if current_logs else None

                if current_hash != last_log_hash and current_logs:
                    last_log_hash = current_hash

                    # Only log new content at debug level
                    self.logger.debug(f"New log content detected for {run_id}, processing {len(current_logs)} chars")

                    # Debug log chunk (rate limited)
                    if DEBUG_METRICS_INGESTION:
                        now = time.time()
                        last = self._debug_last_log_time.get(run_id, 0)
                        if now - last > 30:
                            snippet = current_logs[-500:] if len(current_logs) > 500 else current_logs
                            contains_stats = "[Stats]" in snippet
                            self.logger.warning(f"[METRICS DEBUG] run_id={run_id} log snippet (contains [Stats]={contains_stats}): {snippet[:500]}")
                            self._debug_last_log_time[run_id] = now

                    # Parse metrics from logs
                    target_total = self._get_target_total_timesteps(run_id)
                    metrics_list = self.log_parser.parse_log_chunk(
                        current_logs,
                        run_id,
                        target_total_timesteps=target_total
                    )
                    self.logger.debug(f"Parsed {len(metrics_list)} metrics from logs for {run_id}")

                    # Refresh cached target if the logs revealed it
                    if metrics_list:
                        latest_total = metrics_list[-1].total_timesteps
                        if latest_total and metrics_list[-1].progress_pct > 0:
                            self._target_timesteps_cache[run_id] = latest_total
                    
                    # Add system metrics to latest metric
                    if metrics_list:
                        system_metrics = self.system_monitor.get_system_metrics(pid)
                        latest_metric = metrics_list[-1]

                        # Update with system metrics
                        for key, value in system_metrics.items():
                            setattr(latest_metric, key, value)

                        # Store metrics in database
                        self.logger.debug(f"Storing {len(metrics_list)} metrics in database for {run_id}")
                        for metric in metrics_list:
                            self.database.add_training_metrics(metric)
                            self.logger.debug(f"Saved metric: timestep={metric.timestep}, reward={metric.episode_reward_mean}, fps={metric.fps}")

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
