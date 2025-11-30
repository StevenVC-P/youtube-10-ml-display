"""
Retro ML Desktop - Reinforcement Learning Training Platform

A desktop application for training RL agents on retro games.
"""

__version__ = "0.1.0"
__author__ = "Retro ML Team"

from retro_ml.core.experiments.config import ExperimentConfig, RunType
from retro_ml.core.engine.training_engine import run_experiment, TrainingEngine, ExperimentResult
from retro_ml.core.metrics.event_bus import MetricEventBus, MetricEvent, EventType

__all__ = [
    "ExperimentConfig",
    "RunType",
    "run_experiment",
    "TrainingEngine",
    "ExperimentResult",
    "MetricEventBus",
    "MetricEvent",
    "EventType",
]

