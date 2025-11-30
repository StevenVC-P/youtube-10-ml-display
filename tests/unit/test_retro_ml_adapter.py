"""
Unit tests for the retro_ml adapter layer.

Tests the MetricEventBusAdapter that provides backward compatibility
between the UI's string-based event types and retro_ml's EventType enum.
"""

import pytest
from unittest.mock import Mock, MagicMock
from tools.retro_ml_desktop.retro_ml_adapter import MetricEventBusAdapter
from tools.retro_ml_desktop.metric_event_bus import EventTypes
from retro_ml.core.metrics.event_bus import EventType


class TestMetricEventBusAdapter:
    """Unit tests for MetricEventBusAdapter."""
    
    def test_adapter_has_event_type_map(self):
        """Test that adapter has EVENT_TYPE_MAP defined."""
        assert hasattr(MetricEventBusAdapter, 'EVENT_TYPE_MAP')
        assert isinstance(MetricEventBusAdapter.EVENT_TYPE_MAP, dict)
        assert len(MetricEventBusAdapter.EVENT_TYPE_MAP) > 0
    
    def test_training_started_mapping(self):
        """Test that training.started maps to RUN_STARTED."""
        mapping = MetricEventBusAdapter.EVENT_TYPE_MAP
        assert 'training.started' in mapping
        assert mapping['training.started'] == EventType.RUN_STARTED
    
    def test_training_progress_mapping(self):
        """Test that training.progress maps to TRAINING_STEP."""
        mapping = MetricEventBusAdapter.EVENT_TYPE_MAP
        assert 'training.progress' in mapping
        assert mapping['training.progress'] == EventType.TRAINING_STEP
    
    def test_training_complete_mapping(self):
        """Test that training.complete maps to RUN_COMPLETED."""
        mapping = MetricEventBusAdapter.EVENT_TYPE_MAP
        assert 'training.complete' in mapping
        assert mapping['training.complete'] == EventType.RUN_COMPLETED
    
    def test_training_error_mapping(self):
        """Test that training.error maps to ERROR."""
        mapping = MetricEventBusAdapter.EVENT_TYPE_MAP
        assert 'training.error' in mapping
        assert mapping['training.error'] == EventType.ERROR
    
    def test_training_failed_mapping(self):
        """Test that training.failed maps to ERROR."""
        mapping = MetricEventBusAdapter.EVENT_TYPE_MAP
        assert 'training.failed' in mapping
        assert mapping['training.failed'] == EventType.ERROR
    
    def test_training_stopped_mapping(self):
        """Test that training.stopped maps to RUN_COMPLETED."""
        mapping = MetricEventBusAdapter.EVENT_TYPE_MAP
        assert 'training.stopped' in mapping
        assert mapping['training.stopped'] == EventType.RUN_COMPLETED
    
    def test_training_paused_mapping(self):
        """Test that training.paused maps to TRAINING_STEP."""
        mapping = MetricEventBusAdapter.EVENT_TYPE_MAP
        assert 'training.paused' in mapping
        assert mapping['training.paused'] == EventType.TRAINING_STEP
    
    def test_training_resumed_mapping(self):
        """Test that training.resumed maps to TRAINING_STEP."""
        mapping = MetricEventBusAdapter.EVENT_TYPE_MAP
        assert 'training.resumed' in mapping
        assert mapping['training.resumed'] == EventType.TRAINING_STEP
    
    def test_adapter_subscribe_with_string(self):
        """Test subscribing with string event type."""
        adapter = MetricEventBusAdapter()
        callback = Mock()
        
        # Should not raise an error
        adapter.subscribe('training.started', callback)
    
    def test_adapter_subscribe_with_enum(self):
        """Test subscribing with EventType enum."""
        adapter = MetricEventBusAdapter()
        callback = Mock()
        
        # Should not raise an error
        adapter.subscribe(EventType.RUN_STARTED, callback)
    
    def test_adapter_publish_with_string(self):
        """Test publishing with string event type."""
        adapter = MetricEventBusAdapter()
        callback = Mock()

        adapter.subscribe('training.started', callback)
        adapter.publish('training.started', {'test': 'data'})

        callback.assert_called_once()

    def test_adapter_publish_with_different_string(self):
        """Test publishing with different string event type."""
        adapter = MetricEventBusAdapter()
        callback = Mock()

        adapter.subscribe('training.progress', callback)
        adapter.publish('training.progress', {'test': 'data'})

        callback.assert_called_once()
    
    def test_adapter_unsubscribe_with_string(self):
        """Test unsubscribing with string event type."""
        adapter = MetricEventBusAdapter()
        callback = Mock()

        adapter.subscribe('training.started', callback)
        adapter.unsubscribe('training.started', callback)
        adapter.publish('training.started', {'test': 'data'})

        # Callback should not be called after unsubscribe
        callback.assert_not_called()

    def test_adapter_unsubscribe_with_different_string(self):
        """Test unsubscribing with different string event type."""
        adapter = MetricEventBusAdapter()
        callback = Mock()

        adapter.subscribe('training.progress', callback)
        adapter.unsubscribe('training.progress', callback)
        adapter.publish('training.progress', {'test': 'data'})

        # Callback should not be called after unsubscribe
        callback.assert_not_called()

    def test_adapter_cross_type_subscription(self):
        """Test that different string types map to different events."""
        adapter = MetricEventBusAdapter()
        callback1 = Mock()
        callback2 = Mock()

        # Subscribe to different event types
        adapter.subscribe('training.started', callback1)
        adapter.subscribe('training.progress', callback2)

        # Publish to first event type
        adapter.publish('training.started', {'test': 'data'})

        # Only first callback should be called
        callback1.assert_called_once()
        callback2.assert_not_called()
    
    def test_adapter_handles_unknown_string_event(self):
        """Test that adapter handles unknown string event types gracefully."""
        adapter = MetricEventBusAdapter()
        callback = Mock()
        
        # Subscribe to unknown event type
        # Should not raise an error (might just pass through)
        adapter.subscribe('unknown.event', callback)

