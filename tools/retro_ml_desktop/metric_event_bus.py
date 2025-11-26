"""
Metric Event Bus - Pub/Sub System for Real-Time Updates

Enables decoupled communication between training processes and UI components.
Components can subscribe to events and receive updates without tight coupling.

Usage:
    # Publisher (Process Manager)
    event_bus.publish('training.progress', {
        'experiment_id': 'exp_123',
        'progress_pct': 45.2,
        'timestep': 1000000
    })

    # Subscriber (Dashboard UI)
    def on_progress(data):
        print(f"Training {data['progress_pct']}% complete")

    event_bus.subscribe('training.progress', on_progress)
"""

import threading
from typing import Dict, List, Callable, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricEventBus:
    """
    Pub/Sub event bus for real-time metric and status updates.

    Thread-safe implementation supporting multiple subscribers per event type.
    """

    def __init__(self):
        """Initialize the event bus."""
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 100  # Keep last 100 events for debugging

        logger.info("MetricEventBus initialized")

    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to (e.g., 'training.progress')
            callback: Function to call when event occurs. Receives event data dict.

        Example:
            def on_complete(data):
                print(f"Training {data['experiment_id']} complete!")

            bus.subscribe('training.complete', on_complete)
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type} (total subscribers: {len(self._subscribers[event_type])})")

    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Event type to unsubscribe from
            callback: The callback function to remove

        Returns:
            True if callback was found and removed, False otherwise
        """
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from {event_type}")

                    # Clean up empty subscriber lists
                    if not self._subscribers[event_type]:
                        del self._subscribers[event_type]

                    return True
                except ValueError:
                    logger.warning(f"Callback not found for {event_type}")
                    return False
            return False

    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event (e.g., 'training.progress')
            data: Event data dictionary

        Example:
            bus.publish('training.progress', {
                'experiment_id': 'exp_123',
                'progress_pct': 45.2,
                'timestep': 1000000,
                'metrics': {'reward': 189.5, 'loss': 0.042}
            })
        """
        # Add timestamp to event data
        event_data = {
            **data,
            '_timestamp': datetime.now().isoformat(),
            '_event_type': event_type
        }

        # Store in history (thread-safe)
        with self._lock:
            self._event_history.append(event_data)

            # Trim history if too long
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

            # Get subscribers (copy to avoid lock during callbacks)
            subscribers = self._subscribers.get(event_type, []).copy()

        # Call subscribers outside lock to avoid deadlock
        if subscribers:
            logger.debug(f"Publishing {event_type} to {len(subscribers)} subscribers")

            for callback in subscribers:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback for {event_type}: {e}", exc_info=True)
        else:
            logger.debug(f"No subscribers for {event_type}")

    def get_event_history(self, event_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent event history (for debugging or replay).

        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return

        Returns:
            List of recent events (newest first)
        """
        with self._lock:
            history = self._event_history.copy()

        # Filter by event type if specified
        if event_type:
            history = [e for e in history if e.get('_event_type') == event_type]

        # Return newest first, limited
        return list(reversed(history[-limit:]))

    def clear_subscribers(self, event_type: str = None) -> None:
        """
        Clear all subscribers (useful for testing).

        Args:
            event_type: Optional event type to clear. If None, clears all.
        """
        with self._lock:
            if event_type:
                if event_type in self._subscribers:
                    del self._subscribers[event_type]
                    logger.info(f"Cleared subscribers for {event_type}")
            else:
                self._subscribers.clear()
                logger.info("Cleared all subscribers")

    def get_subscriber_count(self, event_type: str) -> int:
        """
        Get number of subscribers for an event type.

        Args:
            event_type: Event type to check

        Returns:
            Number of subscribers
        """
        with self._lock:
            return len(self._subscribers.get(event_type, []))

    def get_all_event_types(self) -> List[str]:
        """
        Get all event types that have subscribers.

        Returns:
            List of event type strings
        """
        with self._lock:
            return list(self._subscribers.keys())


# Global singleton instance
_global_event_bus = None


def get_event_bus() -> MetricEventBus:
    """
    Get the global event bus singleton.

    Returns:
        MetricEventBus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = MetricEventBus()
    return _global_event_bus


# Standard event types (documentation)
class EventTypes:
    """Standard event types used throughout the application."""

    # Training events
    TRAINING_STARTED = 'training.started'
    TRAINING_PROGRESS = 'training.progress'
    TRAINING_COMPLETE = 'training.complete'
    TRAINING_FAILED = 'training.failed'
    TRAINING_PAUSED = 'training.paused'
    TRAINING_RESUMED = 'training.resumed'
    TRAINING_STOPPED = 'training.stopped'

    # Video events
    VIDEO_GENERATION_STARTED = 'video.generation.started'
    VIDEO_GENERATION_PROGRESS = 'video.generation.progress'
    VIDEO_GENERATED = 'video.generated'
    VIDEO_GENERATION_FAILED = 'video.generation.failed'

    # Experiment events
    EXPERIMENT_CREATED = 'experiment.created'
    EXPERIMENT_UPDATED = 'experiment.updated'
    EXPERIMENT_STATUS_CHANGED = 'experiment.status.changed'

    # System events
    SYSTEM_METRICS_UPDATE = 'system.metrics.update'
    SYSTEM_ERROR = 'system.error'
    SYSTEM_WARNING = 'system.warning'


# Example event data structures (documentation)
"""
Event Data Structures:

training.progress:
{
    'experiment_id': str,
    'progress_pct': float (0-100),
    'timestep': int,
    'total_timesteps': int,
    'elapsed_time': float (seconds),
    'estimated_time_remaining': float (seconds),
    'metrics': {
        'reward_mean': float,
        'loss': float,
        'episode_count': int
    }
}

training.complete:
{
    'experiment_id': str,
    'total_time': float (seconds),
    'final_metrics': {
        'reward_mean': float,
        'reward_max': float,
        'total_episodes': int
    }
}

video.generated:
{
    'experiment_id': str,
    'video_path': str,
    'video_type': str ('milestone', 'hour', 'evaluation'),
    'duration': float (seconds),
    'size_mb': float,
    'metadata': {
        'milestone_pct': int (if milestone video),
        'avg_score': float,
        'max_score': float
    }
}

experiment.status.changed:
{
    'experiment_id': str,
    'old_status': str,
    'new_status': str ('pending', 'running', 'completed', 'failed', 'paused'),
    'message': str (optional)
}
"""
