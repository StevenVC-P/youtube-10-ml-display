"""
Adapter layer for migrating UI code to use retro_ml package.

This module provides backward-compatible wrappers and utilities to help
transition from the old tools/retro_ml_desktop implementations to the new
retro_ml core package.

Usage:
    # Instead of:
    # from tools.retro_ml_desktop.experiment_manager import ExperimentConfig
    
    # Use:
    from tools.retro_ml_desktop.retro_ml_adapter import ExperimentConfig
    
    # Or migrate directly to:
    from retro_ml import ExperimentConfig
"""

import logging
from typing import Dict, Any, Callable, Optional
from pathlib import Path

# Import from new retro_ml package
from retro_ml import (
    ExperimentConfig as CoreExperimentConfig,
    MetricEventBus as CoreMetricEventBus,
    MetricEvent,
    EventType,
    RunType,
    run_experiment,
    ExperimentResult,
)

logger = logging.getLogger(__name__)

# Re-export core classes for backward compatibility
ExperimentConfig = CoreExperimentConfig
RunType = RunType
ExperimentResult = ExperimentResult


class MetricEventBusAdapter:
    """
    Adapter for MetricEventBus to provide backward compatibility.
    
    The old UI code uses string-based event types like 'training.progress'.
    This adapter maps them to the new EventType enum.
    """
    
    # Mapping from old string event types to new EventType enum
    EVENT_TYPE_MAP = {
        'training.started': EventType.RUN_STARTED,
        'training.progress': EventType.TRAINING_STEP,
        'training.step': EventType.TRAINING_STEP,
        'training.episode': EventType.EPISODE_FINISHED,
        'training.complete': EventType.RUN_COMPLETED,
        'training.completed': EventType.RUN_COMPLETED,
        'training.checkpoint': EventType.CHECKPOINT_SAVED,
        'training.error': EventType.ERROR,
        'training.failed': EventType.ERROR,  # Failed training maps to ERROR
        'training.stopped': EventType.RUN_COMPLETED,  # Stopped training maps to RUN_COMPLETED
        'training.paused': EventType.TRAINING_STEP,  # Paused training maps to TRAINING_STEP
        'training.resumed': EventType.TRAINING_STEP,  # Resumed training maps to TRAINING_STEP
        'error': EventType.ERROR,
    }
    
    def __init__(self):
        """Initialize the adapter with a core MetricEventBus."""
        self._core_bus = CoreMetricEventBus()
        self._callback_map: Dict[str, Dict[Callable, Callable]] = {}
        logger.info("MetricEventBusAdapter initialized (wrapping retro_ml.MetricEventBus)")
    
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to an event type (backward compatible).
        
        Args:
            event_type: String event type (e.g., 'training.progress')
            callback: Function to call when event occurs. Receives event data dict.
        """
        # Map string event type to EventType enum
        core_event_type = self.EVENT_TYPE_MAP.get(event_type)
        
        if core_event_type is None:
            logger.warning(f"Unknown event type '{event_type}', mapping to TRAINING_STEP")
            core_event_type = EventType.TRAINING_STEP
        
        # Create wrapper callback that extracts data from MetricEvent
        def wrapper_callback(event: MetricEvent):
            # Extract data dict from MetricEvent for backward compatibility
            callback(event.data)
        
        # Store mapping for unsubscribe
        if event_type not in self._callback_map:
            self._callback_map[event_type] = {}
        self._callback_map[event_type][callback] = wrapper_callback
        
        # Subscribe to core bus
        self._core_bus.subscribe(core_event_type, wrapper_callback)
        logger.debug(f"Subscribed to {event_type} (mapped to {core_event_type})")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event type (backward compatible).
        
        Args:
            event_type: String event type to unsubscribe from
            callback: The callback function to remove
        
        Returns:
            True if callback was found and removed, False otherwise
        """
        core_event_type = self.EVENT_TYPE_MAP.get(event_type, EventType.TRAINING_STEP)
        
        # Get wrapper callback
        if event_type in self._callback_map and callback in self._callback_map[event_type]:
            wrapper_callback = self._callback_map[event_type][callback]
            result = self._core_bus.unsubscribe(core_event_type, wrapper_callback)
            
            # Clean up mapping
            del self._callback_map[event_type][callback]
            if not self._callback_map[event_type]:
                del self._callback_map[event_type]
            
            return result
        
        return False
    
    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publish an event (backward compatible).
        
        Args:
            event_type: String event type (e.g., 'training.progress')
            data: Event data dictionary
        """
        core_event_type = self.EVENT_TYPE_MAP.get(event_type, EventType.TRAINING_STEP)
        
        # Extract experiment_id from data if present
        experiment_id = data.get('experiment_id', 'unknown')
        
        # Create MetricEvent and emit
        event = MetricEvent(
            event_type=core_event_type,
            experiment_id=experiment_id,
            data=data
        )
        
        self._core_bus.emit(event)
    
    def get_history(self, event_type: Optional[str] = None) -> list:
        """Get event history (backward compatible)."""
        if event_type:
            core_event_type = self.EVENT_TYPE_MAP.get(event_type)
            return self._core_bus.get_history(event_type=core_event_type)
        return self._core_bus.get_history()
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._core_bus.clear_history()


# Singleton instance for backward compatibility
_event_bus_instance: Optional[MetricEventBusAdapter] = None


def get_event_bus() -> MetricEventBusAdapter:
    """Get the singleton event bus instance (backward compatible)."""
    global _event_bus_instance
    if _event_bus_instance is None:
        _event_bus_instance = MetricEventBusAdapter()
    return _event_bus_instance

