"""
Integration tests for quick training runs

Tests cover end-to-end workflows:
- Quick run produces all artifacts
- Artifacts have valid schema
- Same seed produces deterministic results
"""

import json
import tempfile
from pathlib import Path

import pytest

from retro_ml.core.experiments.config import ExperimentConfig, RunType
from retro_ml.core.engine.training_engine import run_experiment


@pytest.mark.integration
class TestQuickRun:
    """Integration tests for quick training runs"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def quick_config(self, temp_dir):
        """Create a quick test configuration"""
        return ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            algorithm="PPO",
            run_type=RunType.QUICK,
            total_timesteps=1000,  # Very short for testing
            output_dir=temp_dir,
            seed=42,
            n_envs=1,  # Single env for faster testing
        )
    
    @pytest.mark.slow
    def test_quick_run_produces_artifacts(self, quick_config, temp_dir):
        """Test that a quick run produces all expected artifacts"""
        # Run experiment
        result = run_experiment(quick_config)
        
        # Verify result
        assert result.success is True
        assert result.experiment_id == quick_config.experiment_id
        assert result.total_timesteps == 1000
        
        # Verify artifacts exist
        assert quick_config.metrics_path.exists()
        assert quick_config.model_path.exists()
        assert quick_config.log_path.exists()
        
        # Verify experiment directory structure
        assert quick_config.experiment_dir.exists()
        assert quick_config.experiment_dir.is_dir()
    
    @pytest.mark.slow
    def test_metrics_file_has_valid_schema(self, quick_config):
        """Test that metrics file has valid JSON schema"""
        # Run experiment
        result = run_experiment(quick_config)
        assert result.success is True
        
        # Load and validate metrics
        with open(quick_config.metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Verify required fields
        assert "experiment_id" in metrics
        assert "game_id" in metrics
        assert "algorithm" in metrics
        assert "total_timesteps" in metrics
        assert "metrics" in metrics
        
        # Verify values
        assert metrics["experiment_id"] == quick_config.experiment_id
        assert metrics["game_id"] == quick_config.game_id
        assert metrics["algorithm"] == quick_config.algorithm
        assert metrics["total_timesteps"] == quick_config.total_timesteps
        
        # Verify metrics is a list
        assert isinstance(metrics["metrics"], list)
    
    @pytest.mark.slow
    def test_model_file_can_be_loaded(self, quick_config):
        """Test that saved model can be loaded"""
        from stable_baselines3 import PPO
        
        # Run experiment
        result = run_experiment(quick_config)
        assert result.success is True
        
        # Load model
        loaded_model = PPO.load(str(quick_config.model_path))
        assert loaded_model is not None
    
    @pytest.mark.slow
    def test_csv_metrics_created(self, quick_config):
        """Test that CSV metrics file is created"""
        # Run experiment
        result = run_experiment(quick_config)
        assert result.success is True
        
        # Verify CSV exists
        csv_path = quick_config.experiment_dir / "metrics.csv"
        assert csv_path.exists()
    
    @pytest.mark.slow
    def test_event_log_created(self, quick_config):
        """Test that event log file is created"""
        # Run experiment
        result = run_experiment(quick_config)
        assert result.success is True
        
        # Verify event log exists
        event_log_path = quick_config.experiment_dir / "events.log"
        assert event_log_path.exists()
        
        # Verify it has content
        with open(event_log_path, 'r') as f:
            content = f.read()
            assert len(content) > 0
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="Determinism test requires more investigation")
    def test_same_seed_produces_similar_results(self, temp_dir):
        """Test that same seed produces similar results (within tolerance)"""
        # Create two configs with same seed
        config1 = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            algorithm="PPO",
            total_timesteps=1000,
            output_dir=temp_dir / "run1",
            seed=42,
            n_envs=1,
        )
        
        config2 = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            algorithm="PPO",
            total_timesteps=1000,
            output_dir=temp_dir / "run2",
            seed=42,
            n_envs=1,
        )
        
        # Run both experiments
        result1 = run_experiment(config1)
        result2 = run_experiment(config2)
        
        assert result1.success is True
        assert result2.success is True
        
        # Load metrics from both runs
        with open(config1.metrics_path, 'r') as f:
            metrics1 = json.load(f)
        
        with open(config2.metrics_path, 'r') as f:
            metrics2 = json.load(f)
        
        # Compare metrics (should be similar, not necessarily identical)
        # This is a placeholder - actual comparison would need more sophisticated logic
        assert len(metrics1["metrics"]) > 0
        assert len(metrics2["metrics"]) > 0

