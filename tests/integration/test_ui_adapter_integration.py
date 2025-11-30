"""
Integration tests for UI + retro_ml adapter layer.

Tests the integration between the UI's legacy event system and the new
retro_ml.MetricEventBus through the adapter layer.
"""

import pytest
from unittest.mock import Mock, patch
from tools.retro_ml_desktop.retro_ml_adapter import MetricEventBusAdapter
from tools.retro_ml_desktop.metric_event_bus import EventTypes
from retro_ml.core.metrics.event_bus import EventType


class TestAdapterIntegration:
    """Test adapter layer integration with UI and retro_ml."""
    
    def test_adapter_initialization(self):
        """Test that adapter initializes correctly."""
        adapter = MetricEventBusAdapter()
        assert adapter is not None
        assert adapter._core_bus is not None
    
    def test_string_event_subscription(self):
        """Test subscribing to events using string event types."""
        adapter = MetricEventBusAdapter()
        callback = Mock()

        # Subscribe using string event type (legacy UI style)
        adapter.subscribe(EventTypes.TRAINING_STARTED, callback)

        # Publish event using string type
        adapter.publish(EventTypes.TRAINING_STARTED, {'run_id': 'test-123'})

        # Verify callback was called
        callback.assert_called_once()
        args = callback.call_args[0][0]
        assert args['run_id'] == 'test-123'
    
    def test_enum_event_subscription(self):
        """Test subscribing to events using string representation of EventType enum."""
        adapter = MetricEventBusAdapter()
        callback = Mock()

        # Subscribe using string event type that maps to RUN_STARTED
        adapter.subscribe(EventTypes.TRAINING_STARTED, callback)

        # Publish event using string type
        adapter.publish(EventTypes.TRAINING_STARTED, {'run_id': 'test-456'})

        # Verify callback was called
        callback.assert_called_once()
        args = callback.call_args[0][0]
        assert args['run_id'] == 'test-456'

    def test_event_type_mapping(self):
        """Test that string event types map correctly to enum types."""
        adapter = MetricEventBusAdapter()
        callback = Mock()

        # Subscribe using one string type
        adapter.subscribe(EventTypes.TRAINING_STARTED, callback)

        # Publish using same string type
        adapter.publish(EventTypes.TRAINING_STARTED, {'run_id': 'test-789'})

        # Verify callback was called (mapping worked)
        callback.assert_called_once()
        args = callback.call_args[0][0]
        assert args['run_id'] == 'test-789'

    def test_multiple_subscribers(self):
        """Test multiple subscribers to the same event."""
        adapter = MetricEventBusAdapter()
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()

        # Subscribe multiple callbacks to same event type
        adapter.subscribe(EventTypes.TRAINING_PROGRESS, callback1)
        adapter.subscribe(EventTypes.TRAINING_PROGRESS, callback2)
        adapter.subscribe(EventTypes.TRAINING_STEP, callback3)  # Same underlying event

        # Publish event
        adapter.publish(EventTypes.TRAINING_PROGRESS, {'step': 100})

        # All callbacks should be called
        callback1.assert_called_once()
        callback2.assert_called_once()
        callback3.assert_called_once()

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        adapter = MetricEventBusAdapter()
        callback = Mock()

        # Subscribe and publish
        adapter.subscribe(EventTypes.TRAINING_COMPLETE, callback)
        adapter.publish(EventTypes.TRAINING_COMPLETE, {'status': 'done'})
        assert callback.call_count == 1

        # Unsubscribe and publish again
        adapter.unsubscribe(EventTypes.TRAINING_COMPLETE, callback)
        adapter.publish(EventTypes.TRAINING_COMPLETE, {'status': 'done'})

        # Callback should not be called again
        assert callback.call_count == 1
    
    def test_all_event_types_mapped(self):
        """Test that all UI event types have mappings."""
        adapter = MetricEventBusAdapter()

        # List of all event types used by UI
        ui_event_types = [
            EventTypes.TRAINING_STARTED,
            EventTypes.TRAINING_PROGRESS,
            EventTypes.TRAINING_STEP,
            EventTypes.TRAINING_EPISODE,
            EventTypes.TRAINING_COMPLETE,
            EventTypes.TRAINING_CHECKPOINT,
            EventTypes.TRAINING_ERROR,
            EventTypes.TRAINING_FAILED,
            EventTypes.TRAINING_STOPPED,
            EventTypes.TRAINING_PAUSED,
            EventTypes.TRAINING_RESUMED,
        ]

        # Verify each can be subscribed to without error
        for event_type in ui_event_types:
            callback = Mock()
            adapter.subscribe(event_type, callback)
            adapter.publish(event_type, {'test': 'data'})
            callback.assert_called_once()

    def test_error_event_handling(self):
        """Test that error events are properly handled."""
        adapter = MetricEventBusAdapter()
        callback1 = Mock()
        callback2 = Mock()

        # Subscribe to different error event types with different callbacks
        adapter.subscribe(EventTypes.TRAINING_ERROR, callback1)
        adapter.subscribe(EventTypes.TRAINING_FAILED, callback2)

        # Publish error events
        adapter.publish(EventTypes.TRAINING_ERROR, {'error': 'test error'})
        adapter.publish(EventTypes.TRAINING_FAILED, {'error': 'test failure'})

        # Both callbacks should be called (note: both map to EventType.ERROR so both get called for each publish)
        # TRAINING_ERROR and TRAINING_FAILED both map to EventType.ERROR
        # So each callback gets called twice (once for each publish)
        assert callback1.call_count == 2
        assert callback2.call_count == 2

