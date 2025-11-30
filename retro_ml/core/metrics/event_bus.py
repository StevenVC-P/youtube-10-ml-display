"""
Metric Event Bus - Publish/Subscribe pattern for metrics

This module implements a simple event bus for metrics collection.
Components can publish events (e.g., training steps, episodes) and
subscribers (e.g., metric writers, UI) can listen for these events.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of metric events"""
    TRAINING_STEP = "training_step"
    EPISODE_FINISHED = "episode_finished"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    CHECKPOINT_SAVED = "checkpoint_saved"
    ERROR = "error"


@dataclass
class MetricEvent:
    """
    A metric event containing training information.
    
    Events are published by the training engine and consumed by
    metric writers, UI components, etc.
    """
    event_type: EventType
    experiment_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_type": self.event_type.value,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


class MetricEventBus:
    """
    Event bus for metric collection using publish/subscribe pattern.
    
    The event bus allows decoupling of metric producers (training engine)
    from metric consumers (writers, UI, etc.).
    
    Example:
        >>> bus = MetricEventBus()
        >>> bus.subscribe(EventType.TRAINING_STEP, lambda event: print(event))
        >>> bus.emit(MetricEvent(EventType.TRAINING_STEP, "exp-123", data={"step": 1}))
    """
    
    def __init__(self):
        """Initialize the event bus"""
        self._subscribers: Dict[EventType, List[Callable[[MetricEvent], None]]] = {}
        self._event_history: List[MetricEvent] = []
        self._max_history = 1000  # Keep last 1000 events in memory
        
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[MetricEvent], None]
    ) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Function to call when event is emitted
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type}")
    
    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[MetricEvent], None]
    ) -> None:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type}")
            except ValueError:
                logger.warning(f"Callback not found for {event_type}")
    
    def emit(self, event: MetricEvent) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event: Event to emit
        """
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify subscribers
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(
                        f"Error in event callback for {event.event_type}: {e}",
                        exc_info=True
                    )
    
    def get_history(
        self,
        event_type: Optional[EventType] = None,
        experiment_id: Optional[str] = None
    ) -> List[MetricEvent]:
        """
        Get event history, optionally filtered.
        
        Args:
            event_type: Filter by event type (None = all types)
            experiment_id: Filter by experiment ID (None = all experiments)
            
        Returns:
            List of events matching filters
        """
        events = self._event_history
        
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        
        if experiment_id is not None:
            events = [e for e in events if e.experiment_id == experiment_id]
        
        return events
    
    def clear_history(self) -> None:
        """Clear event history"""
        self._event_history.clear()
        logger.debug("Event history cleared")

