"""
Artifact Writers - Save experiment outputs

This module provides writers for different types of experiment artifacts:
- MetricsWriter: Saves metrics to JSON/CSV
- LogWriter: Saves training logs
- ModelWriter: Saves model checkpoints
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from retro_ml.core.experiments.config import ExperimentConfig
from retro_ml.core.metrics.event_bus import MetricEventBus, MetricEvent, EventType

logger = logging.getLogger(__name__)


class MetricsWriter:
    """
    Writes metrics to JSON and CSV files.
    
    Subscribes to metric events and accumulates them in memory,
    then writes to disk when finalized.
    """
    
    def __init__(self, config: ExperimentConfig, event_bus: MetricEventBus):
        """
        Initialize metrics writer.
        
        Args:
            config: Experiment configuration
            event_bus: Event bus to subscribe to
        """
        self.config = config
        self.event_bus = event_bus
        self.metrics: List[Dict[str, Any]] = []
        
        # Subscribe to relevant events
        self.event_bus.subscribe(EventType.TRAINING_STEP, self._on_training_step)
        self.event_bus.subscribe(EventType.EPISODE_FINISHED, self._on_episode_finished)
        self.event_bus.subscribe(EventType.RUN_COMPLETED, self._on_run_completed)
        
    def _on_training_step(self, event: MetricEvent) -> None:
        """Handle training step event"""
        self.metrics.append({
            "type": "training_step",
            "timestamp": event.timestamp.isoformat(),
            **event.data
        })
    
    def _on_episode_finished(self, event: MetricEvent) -> None:
        """Handle episode finished event"""
        self.metrics.append({
            "type": "episode_finished",
            "timestamp": event.timestamp.isoformat(),
            **event.data
        })
    
    def _on_run_completed(self, event: MetricEvent) -> None:
        """Handle run completed event"""
        self.metrics.append({
            "type": "run_completed",
            "timestamp": event.timestamp.isoformat(),
            **event.data
        })
    
    def finalize(self) -> None:
        """Write accumulated metrics to disk"""
        try:
            # Write JSON
            json_path = self.config.metrics_path
            with open(json_path, 'w') as f:
                json.dump({
                    "experiment_id": self.config.experiment_id,
                    "game_id": self.config.game_id,
                    "algorithm": self.config.algorithm,
                    "total_timesteps": self.config.total_timesteps,
                    "metrics": self.metrics
                }, f, indent=2)
            logger.info(f"Metrics saved to {json_path}")
            
            # Write CSV (simplified version)
            csv_path = self.config.experiment_dir / "metrics.csv"
            if self.metrics:
                with open(csv_path, 'w', newline='') as f:
                    # Get all unique keys from metrics
                    keys = set()
                    for m in self.metrics:
                        keys.update(m.keys())
                    
                    writer = csv.DictWriter(f, fieldnames=sorted(keys))
                    writer.writeheader()
                    writer.writerows(self.metrics)
                logger.info(f"Metrics CSV saved to {csv_path}")
                
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}", exc_info=True)


class LogWriter:
    """
    Writes training logs to file.
    
    Subscribes to all events and writes them to a log file.
    """
    
    def __init__(self, config: ExperimentConfig, event_bus: MetricEventBus):
        """
        Initialize log writer.
        
        Args:
            config: Experiment configuration
            event_bus: Event bus to subscribe to
        """
        self.config = config
        self.event_bus = event_bus
        self.log_entries: List[str] = []
        
        # Subscribe to all event types
        for event_type in EventType:
            self.event_bus.subscribe(event_type, self._on_event)
    
    def _on_event(self, event: MetricEvent) -> None:
        """Handle any event"""
        log_entry = f"[{event.timestamp.isoformat()}] {event.event_type.value}: {event.data}"
        self.log_entries.append(log_entry)
    
    def finalize(self) -> None:
        """Write accumulated logs to disk"""
        try:
            log_path = self.config.experiment_dir / "events.log"
            with open(log_path, 'w') as f:
                f.write('\n'.join(self.log_entries))
            logger.info(f"Event log saved to {log_path}")
        except Exception as e:
            logger.error(f"Failed to write event log: {e}", exc_info=True)


class ModelWriter:
    """
    Writes model checkpoints to disk.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize model writer.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
    
    def save_model(self, model: Any) -> None:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save (Stable-Baselines3 model)
        """
        try:
            model_path = self.config.model_path
            model.save(str(model_path))
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}", exc_info=True)

