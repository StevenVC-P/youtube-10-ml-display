"""
Unit tests for MetricEventBus

Tests cover:
- Event emission and subscription
- Multiple subscribers
- Event history
- Unsubscribe functionality
- Error handling in callbacks
"""

import pytest
from unittest.mock import Mock

from retro_ml.core.metrics.event_bus import MetricEventBus, MetricEvent, EventType


class TestMetricEventBus:
    """Test suite for MetricEventBus"""
    
    @pytest.fixture
    def event_bus(self):
        """Create a test event bus"""
        return MetricEventBus()
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample event"""
        return MetricEvent(
            event_type=EventType.TRAINING_STEP,
            experiment_id="test-exp-123",
            data={"step": 1, "reward": 10.5}
        )
    
    def test_initialization(self, event_bus):
        """Test that event bus initializes correctly"""
        assert event_bus is not None
        assert len(event_bus._subscribers) == 0
        assert len(event_bus._event_history) == 0
    
    def test_subscribe_and_emit(self, event_bus, sample_event):
        """Test basic subscribe and emit functionality"""
        callback = Mock()
        
        event_bus.subscribe(EventType.TRAINING_STEP, callback)
        event_bus.emit(sample_event)
        
        callback.assert_called_once_with(sample_event)
    
    def test_multiple_subscribers(self, event_bus, sample_event):
        """Test that multiple subscribers receive events"""
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()
        
        event_bus.subscribe(EventType.TRAINING_STEP, callback1)
        event_bus.subscribe(EventType.TRAINING_STEP, callback2)
        event_bus.subscribe(EventType.TRAINING_STEP, callback3)
        
        event_bus.emit(sample_event)
        
        callback1.assert_called_once_with(sample_event)
        callback2.assert_called_once_with(sample_event)
        callback3.assert_called_once_with(sample_event)
    
    def test_event_type_filtering(self, event_bus):
        """Test that subscribers only receive events of subscribed type"""
        callback_step = Mock()
        callback_episode = Mock()
        
        event_bus.subscribe(EventType.TRAINING_STEP, callback_step)
        event_bus.subscribe(EventType.EPISODE_FINISHED, callback_episode)
        
        step_event = MetricEvent(
            event_type=EventType.TRAINING_STEP,
            experiment_id="test-123",
            data={"step": 1}
        )
        
        event_bus.emit(step_event)
        
        callback_step.assert_called_once_with(step_event)
        callback_episode.assert_not_called()
    
    def test_unsubscribe(self, event_bus, sample_event):
        """Test unsubscribe functionality"""
        callback = Mock()
        
        event_bus.subscribe(EventType.TRAINING_STEP, callback)
        event_bus.emit(sample_event)
        callback.assert_called_once()
        
        # Unsubscribe and emit again
        event_bus.unsubscribe(EventType.TRAINING_STEP, callback)
        event_bus.emit(sample_event)
        
        # Should still be called only once (from before unsubscribe)
        callback.assert_called_once()
    
    def test_event_history(self, event_bus):
        """Test that events are stored in history"""
        event1 = MetricEvent(
            event_type=EventType.TRAINING_STEP,
            experiment_id="exp-1",
            data={"step": 1}
        )
        event2 = MetricEvent(
            event_type=EventType.TRAINING_STEP,
            experiment_id="exp-1",
            data={"step": 2}
        )
        
        event_bus.emit(event1)
        event_bus.emit(event2)
        
        history = event_bus.get_history()
        assert len(history) == 2
        assert history[0] == event1
        assert history[1] == event2
    
    def test_get_history_filtered_by_event_type(self, event_bus):
        """Test filtering history by event type"""
        step_event = MetricEvent(
            event_type=EventType.TRAINING_STEP,
            experiment_id="exp-1",
            data={"step": 1}
        )
        episode_event = MetricEvent(
            event_type=EventType.EPISODE_FINISHED,
            experiment_id="exp-1",
            data={"episode": 1}
        )
        
        event_bus.emit(step_event)
        event_bus.emit(episode_event)
        
        step_history = event_bus.get_history(event_type=EventType.TRAINING_STEP)
        assert len(step_history) == 1
        assert step_history[0] == step_event
    
    def test_get_history_filtered_by_experiment_id(self, event_bus):
        """Test filtering history by experiment ID"""
        event1 = MetricEvent(
            event_type=EventType.TRAINING_STEP,
            experiment_id="exp-1",
            data={"step": 1}
        )
        event2 = MetricEvent(
            event_type=EventType.TRAINING_STEP,
            experiment_id="exp-2",
            data={"step": 1}
        )
        
        event_bus.emit(event1)
        event_bus.emit(event2)
        
        exp1_history = event_bus.get_history(experiment_id="exp-1")
        assert len(exp1_history) == 1
        assert exp1_history[0] == event1
    
    def test_clear_history(self, event_bus, sample_event):
        """Test clearing event history"""
        event_bus.emit(sample_event)
        assert len(event_bus.get_history()) == 1
        
        event_bus.clear_history()
        assert len(event_bus.get_history()) == 0
    
    def test_callback_error_handling(self, event_bus, sample_event):
        """Test that errors in callbacks don't break event bus"""
        def failing_callback(event):
            raise ValueError("Test error")
        
        working_callback = Mock()
        
        event_bus.subscribe(EventType.TRAINING_STEP, failing_callback)
        event_bus.subscribe(EventType.TRAINING_STEP, working_callback)
        
        # Should not raise exception
        event_bus.emit(sample_event)
        
        # Working callback should still be called
        working_callback.assert_called_once_with(sample_event)
    
    def test_event_to_dict(self, sample_event):
        """Test MetricEvent to_dict method"""
        event_dict = sample_event.to_dict()
        
        assert event_dict["event_type"] == "training_step"
        assert event_dict["experiment_id"] == "test-exp-123"
        assert "timestamp" in event_dict
        assert event_dict["data"]["step"] == 1
        assert event_dict["data"]["reward"] == 10.5

