"""
Integration tests for UI experiment creation and management flow.

Tests the end-to-end flow of creating, managing, and tracking experiments
through the UI using the retro_ml package.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from tools.retro_ml_desktop.experiment_manager import Experiment, ExperimentManager
from retro_ml.core.experiments.config import ExperimentConfig


class TestExperimentFlow:
    """Test experiment creation and management flow."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_experiments.db"
    
    @pytest.fixture
    def experiment_manager(self, temp_db_path):
        """Create an experiment manager with temporary database."""
        return ExperimentManager(db_path=str(temp_db_path))
    
    def test_create_experiment_from_config(self, experiment_manager):
        """Test creating an experiment from ExperimentConfig."""
        # Create a config using retro_ml.ExperimentConfig
        config = ExperimentConfig(
            game_id="ALE/Breakout-v5",
            algorithm="ppo",
            total_timesteps=10000,
            n_envs=4,
            save_frequency=5000
        )
        
        # Create experiment
        experiment = experiment_manager.create_experiment(
            name="Test Experiment",
            game="Breakout",
            algorithm="ppo",
            preset="quick",
            config=config
        )
        
        # Verify experiment was created
        assert experiment is not None
        assert experiment.name == "Test Experiment"
        assert experiment.game == "Breakout"
        assert experiment.algorithm == "ppo"
        assert experiment.config.game_id == "ALE/Breakout-v5"
        assert experiment.config.total_timesteps == 10000
    
    def test_experiment_serialization(self, experiment_manager):
        """Test that experiments can be serialized and deserialized."""
        # Create experiment
        config = ExperimentConfig(
            game_id="ALE/Pong-v5",
            algorithm="ppo",
            total_timesteps=50000
        )
        
        experiment = experiment_manager.create_experiment(
            name="Serialization Test",
            game="Pong",
            algorithm="ppo",
            preset="short",
            config=config
        )
        
        # Convert to dict
        exp_dict = experiment.to_dict()
        
        # Verify dict structure
        assert 'id' in exp_dict
        assert 'config' in exp_dict
        assert exp_dict['name'] == "Serialization Test"
        assert exp_dict['config']['game_id'] == "ALE/Pong-v5"
        
        # Recreate from dict
        restored = Experiment.from_dict(exp_dict)
        
        # Verify restoration
        assert restored.id == experiment.id
        assert restored.name == experiment.name
        assert restored.config.game_id == config.game_id
        assert restored.config.total_timesteps == config.total_timesteps
    
    def test_experiment_status_transitions(self, experiment_manager):
        """Test experiment status transitions."""
        config = ExperimentConfig(
            game_id="ALE/SpaceInvaders-v5",
            algorithm="ppo",
            total_timesteps=100000
        )
        
        experiment = experiment_manager.create_experiment(
            name="Status Test",
            game="SpaceInvaders",
            algorithm="ppo",
            preset="medium",
            config=config
        )
        
        # Initial status should be 'pending'
        assert experiment.status == 'pending'
        
        # Start experiment
        experiment.status = 'running'
        experiment.started = datetime.now()
        assert experiment.status == 'running'
        assert experiment.started is not None
        
        # Complete experiment
        experiment.status = 'completed'
        experiment.completed = datetime.now()
        assert experiment.status == 'completed'
        assert experiment.completed is not None
    
    def test_experiment_with_video_config(self, experiment_manager):
        """Test experiment with video generation configuration."""
        config = ExperimentConfig(
            game_id="ALE/Breakout-v5",
            algorithm="ppo",
            total_timesteps=50000,
            video_length_hours=0.5,  # 30 minutes
            clip_seconds=60,
            milestone_percentages=[25, 50, 75, 100]
        )
        
        experiment = experiment_manager.create_experiment(
            name="Video Test",
            game="Breakout",
            algorithm="ppo",
            preset="custom",
            config=config
        )
        
        # Verify video config
        assert experiment.config.video_length_hours == 0.5
        assert experiment.config.clip_seconds == 60
        assert experiment.config.milestone_percentages == [25, 50, 75, 100]
    
    def test_experiment_with_ppo_hyperparameters(self, experiment_manager):
        """Test experiment with PPO hyperparameters."""
        config = ExperimentConfig(
            game_id="ALE/Pong-v5",
            algorithm="ppo",
            total_timesteps=100000,
            n_steps=256,
            batch_size=512,
            gamma=0.95,
            gae_lambda=0.90,
            ent_coef=0.02,
            vf_coef=0.6,
            max_grad_norm=0.7
        )
        
        experiment = experiment_manager.create_experiment(
            name="Hyperparameter Test",
            game="Pong",
            algorithm="ppo",
            preset="custom",
            config=config
        )
        
        # Verify hyperparameters
        assert experiment.config.n_steps == 256
        assert experiment.config.batch_size == 512
        assert experiment.config.gamma == 0.95
        assert experiment.config.gae_lambda == 0.90
        assert experiment.config.ent_coef == 0.02

