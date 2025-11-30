"""
Pytest configuration and shared fixtures

This module provides common fixtures and configuration for all tests.
"""

import tempfile
from pathlib import Path

import pytest

from retro_ml.core.experiments.config import ExperimentConfig, RunType
from retro_ml.core.metrics.event_bus import MetricEventBus


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def event_bus():
    """Create a fresh MetricEventBus for testing"""
    return MetricEventBus()


@pytest.fixture
def quick_config(temp_dir):
    """Create a quick test configuration"""
    return ExperimentConfig(
        game_id="BreakoutNoFrameskip-v4",
        algorithm="PPO",
        run_type=RunType.QUICK,
        total_timesteps=1000,
        output_dir=temp_dir,
        seed=42,
        n_envs=1,
    )


@pytest.fixture
def sample_configs(temp_dir):
    """Create a set of sample configurations for testing"""
    return {
        "quick": ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            run_type=RunType.QUICK,
            output_dir=temp_dir / "quick",
        ),
        "short": ExperimentConfig(
            game_id="PongNoFrameskip-v4",
            run_type=RunType.SHORT,
            output_dir=temp_dir / "short",
        ),
        "custom": ExperimentConfig(
            game_id="SpaceInvadersNoFrameskip-v4",
            run_type=RunType.CUSTOM,
            total_timesteps=50000,
            output_dir=temp_dir / "custom",
        ),
    }


def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, end-to-end)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (> 1 minute)"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

